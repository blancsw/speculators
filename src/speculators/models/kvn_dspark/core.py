# ruff: noqa: C901, PLR2004, PT018, S101, SLF001

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.attention.flex_attention import create_block_mask
from transformers import PretrainedConfig

if TYPE_CHECKING:
    from collections.abc import Callable

from speculators.losses import (
    LossConfig,
    kl_div_loss,
)
from speculators.losses import (
    eager as eager_losses,
)
from speculators.model import SpeculatorModel
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.kvn_dspark.config import KvNDSparkSpeculatorConfig
from speculators.models.kvn_dspark.losses import (
    compute_kvn_metrics,
    kvn_tv_loss,
    resolve_kvn_loss_config,
)
from speculators.models.kvn_dspark.model_definitions import (
    BornConfidenceHead,
    CausalKvNRFFLayer,
    IntraBlockCausalMixer,
    KrausPOVMHead,
    MultiHorizonKoopmanRollout,
    SpeculativeEngram,
)
from speculators.models.utils import conditional_torch_compile

_HAS_TRITON = importlib.util.find_spec("triton") is not None
_kvn_compile = conditional_torch_compile if _HAS_TRITON else (lambda fn: fn)


def _default_kl_fn(logits: Tensor, targets: Tensor) -> Tensor:
    if _HAS_TRITON and logits.is_cuda:
        return kl_div_loss(logits, targets)
    return eager_losses.kl_div_loss(logits, targets)


_DEFAULT_KL_FN = _default_kl_fn
_DEFAULT_TV_FN = kvn_tv_loss
_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (_DEFAULT_KL_FN, 1.0)}


def kvn_select_anchors(
    loss_mask: Tensor,
    num_anchors: int,
    block_size: int,
    *,
    sample_from_anchor: bool = True,
    document_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Select valid speculative anchors with exact a + 1 target and doc alignment."""
    if not sample_from_anchor:
        cand = (
            loss_mask.bool() & (document_ids != -1)
            if document_ids is not None
            else loss_mask
        )
        return select_anchors(cand, num_anchors, block_size)
    if loss_mask.ndim != 2:
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")
    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")
    cand = torch.zeros_like(loss_mask, dtype=torch.bool)
    if loss_mask.shape[1] > 1:
        cand[:, :-1] = loss_mask[:, 1:].bool()
        if document_ids is not None:
            cand[:, :-1] &= (document_ids[:, :-1] == document_ids[:, 1:]) & (
                document_ids[:, :-1] != -1
            )
    return select_anchors(cand, num_anchors, max(1, block_size - 1))


__all__ = [
    "KvNDSparkDraftModel",
    "kvn_select_anchors",
]


@SpeculatorModel.register("kvn_dspark")
class KvNDSparkDraftModel(DSparkDraftModel):
    """Zero-Drafter-KV-Cache Symplectic-Squeezed Bosonic SSM Speculator Model (v3).

    Supports three controlled scientific variants via ``config.kvn_backbone_type``:
    - ``"kvn"``: 100% Zero-KV-Cache causal driven-damped Bosonic SSM layers (with
      multiplicative dyadic RFF, sp(4, R) butterfly symplectic-squeezed scans, and
      Lindblad purity) + Multi-Horizon Koopman Rollout + IntraBlockCausalMixer +
      SpeculativeEngram + MarkovHead + BornConfidenceHead + KrausPOVMHead.
    - ``"hybrid_1swa_2kvn"``: Sequential (L-1) KvN-RFF layers -> Rollout -> 1 SWA layer.
    - ``"swa"``: Standard DSpark SWA attention stack (baseline).
    """

    config_class: ClassVar[type[KvNDSparkSpeculatorConfig]] = KvNDSparkSpeculatorConfig  # type: ignore[misc,assignment]
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Qwen3DFlashDecoderLayer",
        "CausalKvNRFFLayer",
        "IntraBlockCausalMixer",
    ]

    def _build_kvn_layer(
        self, hidden_size: int, layer_idx: int = 0
    ) -> CausalKvNRFFLayer:
        cfg = self.config
        return CausalKvNRFFLayer(
            d_model=hidden_size,
            n_modes=cfg.kvn_n_modes,
            reupload_layers_l=cfg.kvn_reupload_layers,
            sigma_g=cfg.kvn_sigma_g,
            d_re=cfg.kvn_reupload_dim,
            learnable_rff=cfg.kvn_learnable_rff,
            damping_min=cfg.kvn_damping_min,
            damping_max=cfg.kvn_damping_max,
            enable_interferometer=cfg.kvn_enable_interferometer,
            vacuum_born_norm=cfg.kvn_vacuum_born_norm,
            ffn_mult=cfg.kvn_ffn_mult,
            dyadic_rff=cfg.dyadic_rff,
            use_symplectic_butterfly=cfg.use_symplectic_butterfly,
            squeezing_max_ratio=cfg.squeezing_max_ratio,
            use_lindblad_purity=cfg.use_lindblad_purity,
            use_triton_scan=cfg.use_triton_scan,
            layer_idx=layer_idx,
        )

    def __init__(self, config: KvNDSparkSpeculatorConfig) -> None:
        if (
            config.kvn_backbone_type != "swa"
            and (
                config.use_symplectic_butterfly
                or config.kvn_enable_interferometer
                or config.kvn_vacuum_born_norm
            )
            and config.kvn_n_modes % 2 != 0
        ):
            raise ValueError(
                "kvn_n_modes must be even when dyadic/butterfly modules are active, "
                f"got {config.kvn_n_modes}."
            )

        super().__init__(config=config)

        tl_config = config.transformer_layer_config
        hidden_size = tl_config.hidden_size
        num_draft_layers = tl_config.num_hidden_layers
        n_heads = getattr(tl_config, "num_attention_heads", 8) or 8

        self.kvn_layers = nn.ModuleList()
        self.rollout: MultiHorizonKoopmanRollout | None = None
        self.intrablock_mixer: IntraBlockCausalMixer | None = None

        if config.kvn_backbone_type == "kvn":
            self.layers = nn.ModuleList()
            self.sliding_window_indices = []
            self.uses_sliding_window_attn = False
            self.uses_full_attn = False
            for l_idx in range(num_draft_layers):
                self.kvn_layers.append(self._build_kvn_layer(hidden_size, l_idx))
            self.rollout = MultiHorizonKoopmanRollout(
                d_model=hidden_size,
                n_modes=config.kvn_n_modes,
                block_size=self.block_size,
                anchor_rank=config.kvn_anchor_rank,
                enable_interferometer=config.kvn_enable_interferometer,
                vacuum_born_norm=config.kvn_vacuum_born_norm,
                num_layers=len(self.kvn_layers),
                multilayer_rollout=config.kvn_multilayer_rollout,
                use_lindblad_purity=config.use_lindblad_purity,
            )
            if config.kvn_intrablock_mixer:
                self.intrablock_mixer = IntraBlockCausalMixer(
                    d_model=hidden_size,
                    block_size=self.block_size,
                    n_heads=min(8, n_heads),
                )
        elif config.kvn_backbone_type == "hybrid_1swa_2kvn":
            self.layers = nn.ModuleList(list(self.layers[:1]))
            self.sliding_window_indices = [
                i for i in self.sliding_window_indices if i < 1
            ]
            self.uses_sliding_window_attn = bool(self.sliding_window_indices)
            self.uses_full_attn = bool(
                len(self.layers) - len(self.sliding_window_indices)
            )
            num_kvn = max(1, num_draft_layers - 1)
            for l_idx in range(num_kvn):
                self.kvn_layers.append(self._build_kvn_layer(hidden_size, l_idx))
            self.rollout = MultiHorizonKoopmanRollout(
                d_model=hidden_size,
                n_modes=config.kvn_n_modes,
                block_size=self.block_size,
                anchor_rank=config.kvn_anchor_rank,
                enable_interferometer=config.kvn_enable_interferometer,
                vacuum_born_norm=config.kvn_vacuum_born_norm,
                num_layers=len(self.kvn_layers),
                multilayer_rollout=config.kvn_multilayer_rollout,
                use_lindblad_purity=config.use_lindblad_purity,
            )
            if config.kvn_intrablock_mixer:
                self.intrablock_mixer = IntraBlockCausalMixer(
                    d_model=hidden_size,
                    block_size=self.block_size,
                    n_heads=min(8, n_heads),
                )

        self.engram: SpeculativeEngram | None = None
        if config.use_engram:
            engram_out = (
                config.markov_rank if self.markov_head is not None else hidden_size
            )
            self.engram = SpeculativeEngram(
                out_dim=engram_out,
                engram_buckets=config.engram_buckets,
                engram_dim=config.engram_dim,
            )

        self.kraus_head: KrausPOVMHead | None = None
        if config.use_kraus_tree and self.rollout is not None:
            self.kraus_head = KrausPOVMHead(
                d_model=hidden_size,
                n_modes=config.kvn_n_modes,
                top_c=config.kraus_top_c,
            )

        self.born_confidence_head: BornConfidenceHead | None = None
        if config.enable_confidence_head:
            use_born = config.kvn_use_born_confidence and (self.rollout is not None)
            markov_dim = (
                config.markov_rank
                if (config.confidence_head_with_markov and self.markov_head is not None)
                else 0
            )
            self.born_confidence_head = BornConfidenceHead(
                hidden_size=hidden_size,
                markov_rank=markov_dim,
                n_modes=config.kvn_n_modes,
                use_born=use_born,
                vacuum_born_norm=config.kvn_vacuum_born_norm,
                use_lindblad_purity=config.use_lindblad_purity,
            )
            self.confidence_head = None
        if (
            not _HAS_TRITON
            and getattr(
                config.transformer_layer_config,
                "_attn_implementation",
                "simple_flex_attention",
            )
            == "simple_flex_attention"
        ):
            self._create_mask_fn = create_block_mask
        object.__setattr__(self, "_full_verifier_lm_head_weight", None)
        object.__setattr__(self, "_last_shortlist_stats", (None, None, None))
        self.gradient_checkpointing: bool = False

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse=recurse)
        full_w = getattr(self, "_full_verifier_lm_head_weight", None)
        if isinstance(full_w, Tensor):
            object.__setattr__(
                self,
                "_full_verifier_lm_head_weight",
                fn(full_w).detach().requires_grad_(False),
            )
        return out

    def load_verifier_weights(self) -> None:
        """Load verifier weights and cache non-persistent full lm_head for OOV stats."""
        super().load_verifier_weights()
        spec_cfg = getattr(self.config, "speculators_config", None)
        verifier_path = spec_cfg.verifier.name_or_path if spec_cfg is not None else None
        if self.use_draft_vocab and verifier_path is not None:
            from speculators.utils.loading import load_model_layers  # noqa: PLC0415

            layers = load_model_layers(["lm_head.weight"], verifier_path)
            full_w = layers.get("lm_head.weight", self.embed_tokens.weight)
            full_w = (
                full_w.detach()
                .clone()
                .to(
                    device=self.verifier_lm_head.weight.device,
                    dtype=self.verifier_lm_head.weight.dtype,
                )
            )
            full_w.requires_grad_(False)
            object.__setattr__(self, "_full_verifier_lm_head_weight", full_w)

    def build_custom_optimizers(
        self, config: Any, model: nn.Module | None = None
    ) -> list[torch.optim.Optimizer]:
        """Build BlockwiseMuon + SinkhornAdamW optimizers for KvN-DSpark."""
        from speculators.models.kvn_dspark.optimizers import (  # noqa: PLC0415
            build_kvn_optimizers,
        )

        return build_kvn_optimizers(model if model is not None else self, config)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        super().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        super().gradient_checkpointing_disable()
        self.gradient_checkpointing = False

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: Tensor | None = None,
        d2t: Tensor | None = None,
        **kwargs,
    ) -> KvNDSparkDraftModel:
        """Create a KvNDSparkDraftModel from training CLI/YAML arguments."""

        def _pick(keys: tuple[str, ...], default: Any) -> Any:
            for k in keys:
                if k in kwargs and kwargs[k] is not None:
                    return kwargs[k]
            return default

        def _bool_pick(keys: tuple[str, ...], default: bool) -> bool:
            return bool(_pick(keys, default))

        base_kw = dict(kwargs)
        base_kw["sample_from_anchor"] = _bool_pick(("sample_from_anchor",), True)
        config = KvNDSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("kvn_dspark", verifier_config, **base_kw),
            markov_rank=int(_pick(("markov_rank",), 256)),
            markov_head_type=str(_pick(("markov_head_type",), "vanilla")),
            enable_confidence_head=_bool_pick(("enable_confidence_head",), True),
            confidence_head_with_markov=_bool_pick(
                ("confidence_head_with_markov",), True
            ),
            kvn_backbone_type=_pick(("kvn_backbone_type",), "kvn"),
            kvn_n_modes=int(_pick(("kvn_n_modes",), 256)),
            kvn_reupload_layers=int(_pick(("kvn_reupload_layers",), 2)),
            kvn_reupload_dim=int(_pick(("kvn_reupload_dim",), 512)),
            kvn_learnable_rff=_bool_pick(("kvn_learnable_rff",), True),
            kvn_sigma_g=float(_pick(("kvn_sigma_g",), 1.0)),
            kvn_lasso_weight=float(_pick(("kvn_lasso_weight",), 1e-4)),
            kvn_use_born_confidence=_bool_pick(("kvn_use_born_confidence",), True),
            kvn_enable_interferometer=_bool_pick(("kvn_enable_interferometer",), True),
            kvn_vacuum_born_norm=_bool_pick(("kvn_vacuum_born_norm",), True),
            kvn_anchor_rank=int(_pick(("kvn_anchor_rank",), 512)),
            kvn_multilayer_rollout=_bool_pick(("kvn_multilayer_rollout",), True),
            kvn_intrablock_mixer=_bool_pick(("kvn_intrablock_mixer",), True),
            kvn_strict_causal_shift=_bool_pick(("kvn_strict_causal_shift",), True),
            kvn_ffn_mult=float(_pick(("kvn_ffn_mult",), 1.5)),
            kvn_state_dtype=_pick(("kvn_state_dtype",), "float32"),
            dyadic_rff=_bool_pick(("dyadic_rff", "kvn_dyadic_rff"), True),
            use_engram=_bool_pick(("use_engram", "kvn_use_engram"), True),
            engram_buckets=int(_pick(("engram_buckets", "kvn_engram_buckets"), 65536)),
            engram_dim=int(_pick(("engram_dim", "kvn_engram_dim"), 128)),
            use_symplectic_butterfly=_bool_pick(
                (
                    "use_symplectic_butterfly",
                    "kvn_symplectic_butterfly",
                    "kvn_use_symplectic_butterfly",
                ),
                True,
            ),
            squeezing_max_ratio=float(
                _pick(("squeezing_max_ratio", "kvn_squeezing_max_ratio"), 0.8)
            ),
            use_lindblad_purity=_bool_pick(
                (
                    "use_lindblad_purity",
                    "kvn_lindblad_purity",
                    "kvn_use_lindblad_purity",
                ),
                True,
            ),
            use_kraus_tree=_bool_pick(("use_kraus_tree", "kvn_use_kraus_tree"), True),
            kraus_top_c=int(_pick(("kraus_top_c", "kvn_kraus_top_c"), 256)),
            use_triton_scan=_bool_pick(
                ("use_triton_scan", "kvn_use_triton_scan"), True
            ),
            self_cond_rollout_prob=float(
                _pick(
                    (
                        "self_cond_rollout_prob",
                        "kvn_self_cond_prob",
                        "kvn_self_cond_rollout_prob",
                    ),
                    0.0,
                )
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve KvN-DSpark compound loss, falling back when Triton is absent."""
        implementation = kwargs.get("loss_implementation", "fused")
        loss_config = resolve_kvn_loss_config(kwargs["loss_fn"], implementation)
        tv_loss_fn = resolve_kvn_loss_config("tv", implementation)["tv"][0]
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "tv_loss_fn": tv_loss_fn,
            "loss_implementation": implementation,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        train_kwargs = {**shared, "kvn_lasso_weight": kwargs.get("kvn_lasso_weight")}
        val_kwargs = {**shared, "kvn_lasso_weight": 0.0}
        return train_kwargs, val_kwargs

    def total_lasso_penalty(self, lasso_weight: float | None = None) -> Tensor:
        """Compute dyadic Group-LASSO + L1 penalty across all KvN-RFF readouts."""
        eff_weight = (
            self.config.kvn_lasso_weight
            if lasso_weight is None
            else float(lasso_weight)
        )
        if not self.kvn_layers or eff_weight <= 0.0:
            return torch.tensor(0.0, device=self.fc.weight.device)
        pen = sum(layer.lasso_penalty() for layer in self.kvn_layers)
        if self.rollout is not None:
            pen = pen + self.rollout.lasso_penalty()
        if self.born_confidence_head is not None:
            pen = pen + self.born_confidence_head.lasso_penalty()
        return eff_weight * pen

    @torch.no_grad()
    def _compute_joint_dyad_rms_norms(self) -> Tensor:
        """Compute exact O(1) RMS-normalized L2 group norm per dyad across all modal
        weights.
        """
        m = self.kvn_layers[0].n_modes
        n_dyads = m // 2
        dev = self.kvn_layers[0].readout_linear.weight.device
        sum_sq = torch.zeros(n_dyads, dtype=torch.float32, device=dev)
        elem_count = 0

        for layer in self.kvn_layers:
            w = layer.readout_linear.weight.float()
            d_out = w.size(0)
            w_dyad = torch.stack(
                [
                    w[:, :m].view(d_out, n_dyads, 2),
                    w[:, m : 2 * m].view(d_out, n_dyads, 2),
                    w[:, 2 * m :].view(d_out, n_dyads, 2),
                ],
                dim=0,
            )
            sum_sq.add_(w_dyad.square().sum(dim=(0, 1, 3)))
            elem_count += 6 * d_out

            sum_sq.add_(layer.in_gain_q.float().view(n_dyads, 2).square().sum(dim=-1))
            sum_sq.add_(layer.in_gain_p.float().view(n_dyads, 2).square().sum(dim=-1))
            elem_count += 4

            if layer.butterfly_couple_gate is not None:
                sum_sq.add_(
                    layer.butterfly_couple_gate.float()
                    .view(n_dyads, 2)
                    .square()
                    .sum(dim=-1)
                )
                elem_count += 2

            if layer.lindblad_gain is not None:
                sum_sq.add_(
                    layer.lindblad_gain.float().view(n_dyads, 2).square().sum(dim=-1)
                )
                elem_count += 2

        if self.rollout is not None:
            wr = self.rollout.obs_linear.weight.float()
            dr = wr.size(0)
            wr_dyad = torch.stack(
                [
                    wr[:, :m].view(dr, n_dyads, 2),
                    wr[:, m : 2 * m].view(dr, n_dyads, 2),
                    wr[:, 2 * m :].view(dr, n_dyads, 2),
                ],
                dim=0,
            )
            sum_sq.add_(wr_dyad.square().sum(dim=(0, 1, 3)))
            elem_count += 6 * dr

            wf = self.rollout.horizon_force_proj.weight.float().view(
                2 * self.block_size, n_dyads, 2, -1
            )
            bf = self.rollout.horizon_force_proj.bias.float().view(
                2 * self.block_size, n_dyads, 2
            )
            sum_sq.add_(wf.square().sum(dim=(0, 2, 3)))
            sum_sq.add_(bf.square().sum(dim=(0, 2)))
            elem_count += 4 * self.block_size * (wf.size(-1) + 1)

        if (
            self.born_confidence_head is not None
            and self.born_confidence_head.born_proj is not None
        ):
            wb = self.born_confidence_head.born_proj.weight.float()
            sum_sq.add_(wb.view(wb.size(0), n_dyads, 2).square().sum(dim=(0, 2)))
            elem_count += 2 * wb.size(0)

            if self.born_confidence_head.vacuum_pvac_proj is not None:
                wpv = self.born_confidence_head.vacuum_pvac_proj.weight.float()
                sum_sq.add_(wpv.square().sum(dim=0))
                elem_count += wpv.size(0)

            if self.born_confidence_head.lindblad_purity_proj is not None:
                wpu = self.born_confidence_head.lindblad_purity_proj.weight.float()
                sum_sq.add_(wpu.view(wpu.size(0), n_dyads, 2).square().sum(dim=(0, 2)))
                elem_count += 2 * wpu.size(0)

        if self.kraus_head is not None:
            wk = self.kraus_head.kraus_backaction_proj.weight.float()
            dk = wk.size(0)
            wk_dyad = torch.stack(
                [
                    wk[:, :m].view(dk, n_dyads, 2),
                    wk[:, m : 2 * m].view(dk, n_dyads, 2),
                ],
                dim=0,
            )
            sum_sq.add_(wk_dyad.square().sum(dim=(0, 1, 3)))
            sum_sq.add_(
                self.kraus_head.kraus_mode_gate.float()
                .view(n_dyads, 2)
                .square()
                .sum(dim=-1)
            )
            elem_count += 4 * dk + 2

        return torch.sqrt(sum_sq / float(max(1, elem_count)))

    @torch.no_grad()
    def apply_proximal_dyadic_group_lasso(self, step_lambda: float) -> dict[str, float]:
        """Apply exact proximal block soft-thresholding across all dyad modal weights
        in-place.

        Uses dimension-normalized RMS group norm per dyad so ``step_lambda = lr *
        lasso_weight``
        remains O(1)-calibrated across arbitrary ``d_model``. When ``step_lambda >=
        rms_norm[g]``,
        dyad ``g`` is driven to exact IEEE-754 ``0.0`` across all layers and heads
        simultaneously.
        """
        if not self.kvn_layers or self.kvn_layers[0].n_modes % 2 != 0:
            return {
                "zero_dyad_fraction": 0.0,
                "min_dyad_norm": 0.0,
                "max_dyad_norm": 0.0,
            }

        rms_norm = self._compute_joint_dyad_rms_norms()
        if step_lambda <= 0.0:
            return {
                "zero_dyad_fraction": float((rms_norm == 0.0).float().mean().item()),
                "min_dyad_norm": float(rms_norm.min().item()),
                "max_dyad_norm": float(rms_norm.max().item()),
            }

        scale_dyad = torch.clamp(1.0 - float(step_lambda) / (rms_norm + 1e-12), min=0.0)
        scale_mode = scale_dyad.repeat_interleave(2)

        for layer in self.kvn_layers:
            sm = scale_mode.to(
                device=layer.readout_linear.weight.device,
                dtype=layer.readout_linear.weight.dtype,
            )
            s3 = torch.cat([sm, sm, sm], dim=0).unsqueeze(0)
            layer.readout_linear.weight.mul_(s3)
            layer.in_gain_q.mul_(
                scale_mode.to(layer.in_gain_q.device, layer.in_gain_q.dtype)
            )
            layer.in_gain_p.mul_(
                scale_mode.to(layer.in_gain_p.device, layer.in_gain_p.dtype)
            )
            if layer.butterfly_couple_gate is not None:
                layer.butterfly_couple_gate.mul_(
                    scale_mode.to(
                        layer.butterfly_couple_gate.device,
                        layer.butterfly_couple_gate.dtype,
                    )
                )
            if layer.lindblad_gain is not None:
                layer.lindblad_gain.mul_(
                    scale_mode.to(layer.lindblad_gain.device, layer.lindblad_gain.dtype)
                )

        if self.rollout is not None:
            sm_r = scale_mode.to(
                device=self.rollout.obs_linear.weight.device,
                dtype=self.rollout.obs_linear.weight.dtype,
            )
            self.rollout.obs_linear.weight.mul_(
                torch.cat([sm_r, sm_r, sm_r], dim=0).unsqueeze(0)
            )
            sf = sm_r.repeat(2 * self.block_size)
            self.rollout.horizon_force_proj.weight.mul_(sf.unsqueeze(-1))
            self.rollout.horizon_force_proj.bias.mul_(sf)

        if (
            self.born_confidence_head is not None
            and self.born_confidence_head.born_proj is not None
        ):
            bch = self.born_confidence_head
            assert bch.born_proj is not None
            sm_b = scale_mode.to(
                bch.born_proj.weight.device, bch.born_proj.weight.dtype
            )
            bch.born_proj.weight.mul_(sm_b.unsqueeze(0))
            if bch.vacuum_pvac_proj is not None:
                sd_b = scale_dyad.to(
                    bch.vacuum_pvac_proj.weight.device,
                    bch.vacuum_pvac_proj.weight.dtype,
                )
                bch.vacuum_pvac_proj.weight.mul_(sd_b.unsqueeze(0))
            if bch.lindblad_purity_proj is not None:
                bch.lindblad_purity_proj.weight.mul_(sm_b.unsqueeze(0))

        if self.kraus_head is not None:
            sm_k = scale_mode.to(
                self.kraus_head.kraus_backaction_proj.weight.device,
                self.kraus_head.kraus_backaction_proj.weight.dtype,
            )
            self.kraus_head.kraus_backaction_proj.weight.mul_(
                torch.cat([sm_k, sm_k], dim=0).unsqueeze(0)
            )
            self.kraus_head.kraus_mode_gate.mul_(
                scale_mode.to(
                    self.kraus_head.kraus_mode_gate.device,
                    self.kraus_head.kraus_mode_gate.dtype,
                )
            )

        post_norm = rms_norm * scale_dyad
        return {
            "zero_dyad_fraction": float((post_norm == 0.0).float().mean().item()),
            "min_dyad_norm": float(post_norm.min().item()),
            "max_dyad_norm": float(post_norm.max().item()),
        }

    def _resolve_proximal_lasso_weight(
        self, train_call_kwargs: dict | None, kvn_lasso_weight: float | None
    ) -> float:
        if kvn_lasso_weight is not None:
            return float(kvn_lasso_weight)
        train_kw_lasso = (train_call_kwargs or {}).get("kvn_lasso_weight")
        if train_kw_lasso is not None:
            return float(train_kw_lasso)
        return float(getattr(self.config, "kvn_lasso_weight", 0.0) or 0.0)

    def prepare_train_call_kwargs(
        self, train_call_kwargs: dict | None, kvn_lasso_weight: float | None = None
    ) -> dict:
        forward_kwargs = dict(train_call_kwargs or {})
        if (
            self._resolve_proximal_lasso_weight(train_call_kwargs, kvn_lasso_weight)
            > 0.0
        ):
            forward_kwargs["kvn_lasso_weight"] = 0.0
        return forward_kwargs

    def post_optimizer_step(
        self,
        optimizers: list[torch.optim.Optimizer],
        loss: Tensor,
        metrics: dict[str, Tensor],
        train_call_kwargs: dict | None = None,
        kvn_lasso_weight: float | None = None,
    ) -> None:
        prox_weight = self._resolve_proximal_lasso_weight(
            train_call_kwargs, kvn_lasso_weight
        )
        if prox_weight <= 0.0:
            return
        prox_lr = next(
            (
                float(opt.param_groups[0]["lr"])
                for opt in optimizers
                if isinstance(opt, torch.optim.AdamW)
            ),
            float(optimizers[0].param_groups[0]["lr"]),
        )
        prox_stats = self.apply_proximal_dyadic_group_lasso(prox_lr * prox_weight)
        if isinstance(prox_stats, dict) and "zero_dyad_fraction" in prox_stats:
            metrics["zero_dyad_fraction_sum"] = torch.tensor(
                float(prox_stats["zero_dyad_fraction"]),
                dtype=torch.float32,
                device=loss.device,
            )
            metrics["zero_dyad_fraction_total"] = torch.ones(
                (), dtype=torch.float32, device=loss.device
            )

    @torch.no_grad()
    def prune_kvn_modes(self, target_modes: int) -> None:
        """Structurally prune all CausalKvNRFFLayer blocks (M_train -> M_infer)."""
        if not self.kvn_layers:
            return
        old_m = self.kvn_layers[0].n_modes
        if target_modes <= 0 or target_modes >= old_m:
            return
        if any(layer._has_dyadic_structure() for layer in self.kvn_layers) and (
            target_modes % 2 != 0
        ):
            raise ValueError(
                "target_modes must be even when dyadic modules are active, "
                f"got {target_modes}."
            )

        head_imp: Tensor | None = None
        if self.rollout is not None:
            head_imp = self.rollout.mode_importance()
        if self.born_confidence_head is not None:
            born_imp = self.born_confidence_head.mode_importance()
            if born_imp is not None:
                head_imp = (
                    born_imp
                    if head_imp is None
                    else head_imp + born_imp.to(head_imp.device)
                )

        if self.rollout is not None and self.rollout.multilayer_rollout:
            joint_extra: Tensor | None = head_imp
            for layer in self.kvn_layers[:-1]:
                l_imp = layer.mode_importance()
                joint_extra = (
                    l_imp
                    if joint_extra is None
                    else joint_extra + l_imp.to(joint_extra.device)
                )
            shared_top_idx = self.kvn_layers[-1].select_top_indices(
                target_modes, extra_importance=joint_extra
            )
            for layer in self.kvn_layers:
                layer.prune_to_indices(shared_top_idx)
            last_top_idx: Tensor | None = shared_top_idx
        else:
            last_top_idx = None
            num_kvn = len(self.kvn_layers)
            for idx, layer in enumerate(self.kvn_layers):
                extra_imp = head_imp if idx == num_kvn - 1 else None
                last_top_idx = layer.prune_to_top_modes(
                    target_modes, extra_importance=extra_imp
                )

        if last_top_idx is not None:
            if self.rollout is not None:
                self.rollout.prune_to_indices(last_top_idx)
            if self.born_confidence_head is not None:
                self.born_confidence_head.prune_born_modes(last_top_idx)
            if self.kraus_head is not None:
                self.kraus_head.prune_to_indices(last_top_idx)
        self.config.kvn_n_modes = target_modes
        self.config.kvn_lasso_weight = 0.0

    def _resolve_fc_output(self, fc_or_hidden: Tensor) -> Tensor:
        """Accept either projected fc_output [B, T, D] or concatenated aux hidden_states
        [B, T, K*D].
        """
        hidden_size = self.config.transformer_layer_config.hidden_size
        if fc_or_hidden.size(-1) == hidden_size:
            return fc_or_hidden
        return self.hidden_norm(self.fc(fc_or_hidden))

    def _prepare_causal_kvn_inputs(
        self,
        fc_output: Tensor,
        tok_emb: Tensor,
        document_ids: Tensor | None,
        prev_last_fc: Tensor | None = None,
        prev_last_doc_id: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Construct strictly causal x_t = fc_shifted_t + tok_emb_t and reset mask."""
        bsz, seq_len, _ = fc_output.shape
        reset_mask = torch.zeros(
            (bsz, seq_len), dtype=torch.bool, device=fc_output.device
        )
        if prev_last_fc is None:
            reset_mask[:, 0] = True
        elif document_ids is not None and prev_last_doc_id is not None:
            reset_mask[:, 0] = (
                document_ids[:, 0] != prev_last_doc_id.to(document_ids.device).view(bsz)
            ) | (document_ids[:, 0] == -1)

        if document_ids is not None and seq_len > 1:
            reset_mask[:, 1:] = (document_ids[:, 1:] != document_ids[:, :-1]) | (
                document_ids[:, 1:] == -1
            )

        if not self.config.kvn_strict_causal_shift:
            return fc_output + tok_emb, reset_mask

        fc_shifted = torch.zeros_like(fc_output)
        if prev_last_fc is not None and seq_len >= 1:
            fc_shifted[:, :1, :] = prev_last_fc.to(
                device=fc_output.device, dtype=fc_output.dtype
            )
        if seq_len > 1:
            fc_shifted[:, 1:, :] = fc_output[:, :-1, :]
        fc_shifted = fc_shifted.masked_fill(reset_mask.unsqueeze(-1), 0.0)
        return fc_shifted + tok_emb, reset_mask

    @torch.compiler.disable
    def _select_kvn_anchors(
        self,
        loss_mask: Tensor,
        max_anchors: int,
        document_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Select valid anchors outside torch.compile with exact a+1 alignment under
        sample_from_anchor.
        """
        return kvn_select_anchors(
            loss_mask,
            max_anchors,
            self.block_size,
            sample_from_anchor=self.config.sample_from_anchor,
            document_ids=document_ids,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        total_seq_len = loss_mask.shape[1]
        anchor_positions, anchor_valid = self._select_kvn_anchors(
            loss_mask, max_anchors, document_ids=document_ids
        )
        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )
        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
            )
        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    @torch.compiler.disable
    @torch.no_grad()
    def compute_full_verifier_target_stats(
        self,
        normed_hidden: Tensor,
        shortlist_logits: Tensor | None = None,
        chunk_tokens: int = 512,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Compute exact (target_lse_full, in_vocab_mass, target_argmax_in_vocab) over
        V_full.
        """
        full_w = getattr(self, "_full_verifier_lm_head_weight", None)
        if (
            not self.use_draft_vocab
            or not isinstance(full_w, Tensor)
            or self.t2d is None
        ):
            return None
        if full_w.device != normed_hidden.device or full_w.dtype != normed_hidden.dtype:
            full_w = full_w.to(device=normed_hidden.device, dtype=normed_hidden.dtype)
            object.__setattr__(self, "_full_verifier_lm_head_weight", full_w)

        flat_h = normed_hidden.reshape(-1, normed_hidden.shape[-1])
        t2d_dev = self.t2d.to(device=normed_hidden.device, dtype=torch.bool)
        lse_parts: list[Tensor] = []
        argmax_in_parts: list[Tensor] = []
        for start in range(0, flat_h.shape[0], chunk_tokens):
            logits_c = F.linear(flat_h[start : start + chunk_tokens], full_w).float()
            lse_parts.append(torch.logsumexp(logits_c, dim=-1))
            argmax_in_parts.append(t2d_dev[logits_c.argmax(dim=-1)])

        lse_full = torch.cat(lse_parts, dim=0).view(normed_hidden.shape[:-1])
        argmax_in = torch.cat(argmax_in_parts, dim=0).view(normed_hidden.shape[:-1])
        if shortlist_logits is None:
            shortlist_logits = self.verifier_lm_head(normed_hidden)
        shortlist_lse = torch.logsumexp(shortlist_logits.float(), dim=-1)
        in_vocab_mass = torch.exp(shortlist_lse - lse_full).clamp(0.0, 1.0)
        return lse_full, in_vocab_mass, argmax_in

    def _compute_targets_and_mask(
        self,
        verifier_last_hidden_states: Tensor,
        loss_mask: Tensor,
        document_ids: Tensor,
        anchor_positions: Tensor,
        anchor_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute verifier logit targets, OOV stats, and aligned slot loss mask."""
        total_seq_len = verifier_last_hidden_states.shape[1]
        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )

        with torch.no_grad():
            if anchored_block_indices.numel() < total_seq_len:
                target_indices = (
                    anchored_block_indices
                    if self.config.sample_from_anchor
                    else (anchored_block_indices - 1) % total_seq_len
                )
                normed_sub = self.verifier_norm(
                    verifier_last_hidden_states[:, target_indices]
                )
                targets = self.verifier_lm_head(normed_sub)
                stats = self.compute_full_verifier_target_stats(
                    normed_sub, shortlist_logits=targets
                )
                object.__setattr__(
                    self,
                    "_last_shortlist_stats",
                    stats if stats is not None else (None, None, None),
                )
            else:
                normed_all = self.verifier_norm(verifier_last_hidden_states)
                raw_verifier_logits = self.verifier_lm_head(normed_all)
                stats_all = self.compute_full_verifier_target_stats(
                    normed_all, shortlist_logits=raw_verifier_logits
                )
                if not self.config.sample_from_anchor:
                    verifier_logits = torch.roll(raw_verifier_logits, 1, dims=1)
                    stats = (
                        tuple(
                            torch.roll(s, 1, dims=1)[:, anchored_block_indices]
                            for s in stats_all
                        )
                        if stats_all is not None
                        else None
                    )
                else:
                    verifier_logits = raw_verifier_logits
                    stats = (
                        tuple(s[:, anchored_block_indices] for s in stats_all)
                        if stats_all is not None
                        else None
                    )
                targets = verifier_logits[:, anchored_block_indices]
                object.__setattr__(
                    self,
                    "_last_shortlist_stats",
                    stats if stats is not None else (None, None, None),
                )

        valid_block_mask = anchor_valid.repeat_interleave(self.block_size).unsqueeze(0)
        if self.config.sample_from_anchor:
            next_indices = anchored_block_indices + 1
            in_bounds = (next_indices < total_seq_len).unsqueeze(0)
            clamped_next = next_indices.clamp(max=total_seq_len - 1)
            anchor_docs = document_ids[:, anchor_positions].repeat_interleave(
                self.block_size, dim=1
            )
            same_doc = (document_ids[:, clamped_next] == anchor_docs) & (
                anchor_docs != -1
            )
            slot_valid = valid_block_mask & in_bounds & same_doc
            aligned_loss_mask = loss_mask[:, clamped_next].clone() * slot_valid.to(
                loss_mask.dtype
            )
        else:
            same_doc_mask = document_ids[:, anchored_block_indices] == document_ids[
                :, anchor_positions
            ].repeat_interleave(self.block_size, dim=1)
            aligned_loss_mask = (
                loss_mask.clone()[:, anchored_block_indices]
                * same_doc_mask.to(loss_mask.dtype)
                * valid_block_mask.to(loss_mask.dtype)
            )
            aligned_loss_mask[:, :: self.block_size] = 0

        return targets, aligned_loss_mask, anchored_block_indices

    def _kvn_backbone_forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        loss_mask: Tensor,
        verifier_last_hidden_states: Tensor,
        document_ids: Tensor,
        position_ids: Tensor | None = None,
        max_anchors: int = 512,
        **kwargs,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
    ]:
        """Run Pure-KvN, Sequential Hybrid B3, or SWA backbone."""
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if self.config.kvn_backbone_type == "swa":
            if position_ids is None:
                position_ids = torch.arange(
                    total_seq_len, dtype=torch.long, device=device
                ).unsqueeze(0)
            full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
                self._build_attention_mask(loss_mask, max_anchors, document_ids, device)
            )
            mask_tokens_size = max_anchors * self.block_size
            mask_token_ids = torch.full(
                (1, mask_tokens_size),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
            noise_embedding = self.embed_tokens(mask_token_ids)
            fc_output = self.hidden_norm(self.fc(hidden_states))
            mask_position_ids = get_base_indices_for_anchored_blocks(
                position_ids[0, anchor_positions], self.block_size
            )
            pos_ids_cat = torch.cat(
                [position_ids, mask_position_ids.unsqueeze(0)], dim=1
            )
            position_embeddings = self.rotary_emb(hidden_states, pos_ids_cat)
            targets, aligned_loss_mask, anchored_block_indices = (
                self._compute_targets_and_mask(
                    verifier_last_hidden_states=verifier_last_hidden_states,
                    loss_mask=loss_mask,
                    document_ids=document_ids,
                    anchor_positions=anchor_positions,
                    anchor_valid=anchor_valid,
                )
            )
            for layer_idx, layer in enumerate(self.layers):
                noise_embedding = layer(
                    hidden_states=noise_embedding,
                    target_hidden=fc_output,
                    attention_mask=sliding_window_attn_mask
                    if layer_idx in self.sliding_window_indices
                    else full_attn_mask,
                    position_ids=pos_ids_cat,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden = self.norm(noise_embedding)
            logits = self.lm_head(hidden)
            return (
                hidden,
                logits,
                targets,
                aligned_loss_mask,
                anchored_block_indices,
                None,
            )

        is_hybrid = self.config.kvn_backbone_type == "hybrid_1swa_2kvn"

        if is_hybrid:
            if position_ids is None:
                position_ids = torch.arange(
                    total_seq_len, dtype=torch.long, device=device
                ).unsqueeze(0)
            full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
                self._build_attention_mask(loss_mask, max_anchors, document_ids, device)
            )
        else:
            full_attn_mask = None
            sliding_window_attn_mask = None
            anchor_positions, anchor_valid = self._select_kvn_anchors(
                loss_mask, max_anchors, document_ids=document_ids
            )

        fc_output = self.hidden_norm(self.fc(hidden_states))
        targets, aligned_loss_mask, anchored_block_indices = (
            self._compute_targets_and_mask(
                verifier_last_hidden_states=verifier_last_hidden_states,
                loss_mask=loss_mask,
                document_ids=document_ids,
                anchor_positions=anchor_positions,
                anchor_valid=anchor_valid,
            )
        )

        tok_emb = self.embed_tokens(input_ids).to(fc_output.dtype)
        x, reset_mask = self._prepare_causal_kvn_inputs(
            fc_output=fc_output,
            tok_emb=tok_emb,
            document_ids=document_ids,
        )

        q_anchors_per_layer: list[Tensor] = []
        p_anchors_per_layer: list[Tensor] = []
        dtheta_anchors_per_layer: list[Tensor] = []
        nu_anchors_per_layer: list[Tensor] = []

        use_manual_ckpt = self.training and bool(
            getattr(self, "gradient_checkpointing", False)
        )
        for kvn_layer in self.kvn_layers:
            if use_manual_ckpt and not getattr(
                kvn_layer, "gradient_checkpointing", False
            ):
                x, q_seq, p_seq, dtheta_seq, nu_seq = torch.utils.checkpoint.checkpoint(
                    kvn_layer.__call__,
                    x,
                    reset_mask,
                    None,
                    torch.float32,
                    True,
                    use_reentrant=False,
                )
            else:
                x, q_seq, p_seq, dtheta_seq, nu_seq = kvn_layer(
                    x, reset_mask=reset_mask, return_nu=True
                )
            q_anchors_per_layer.append(q_seq[0, anchor_positions])
            p_anchors_per_layer.append(p_seq[0, anchor_positions])
            dtheta_anchors_per_layer.append(dtheta_seq[0, anchor_positions])
            nu_anchors_per_layer.append(nu_seq[0, anchor_positions])

        assert self.rollout is not None and len(q_anchors_per_layer) > 0
        anchor_q, anchor_p, anchor_dtheta, anchor_nu = (
            self.rollout.aggregate_multilayer_states(
                q_anchors_per_layer,
                p_anchors_per_layer,
                dtheta_anchors_per_layer,
                nu_anchors_per_layer,
            )
        )
        anchor_h = x[0, anchor_positions]
        hidden_blocks, born_obs, p_vac, purity, q_obs, p_obs = self.rollout(
            anchor_h=anchor_h,
            anchor_q=anchor_q,
            anchor_p=anchor_p,
            anchor_dtheta=anchor_dtheta,
            anchor_nu=anchor_nu,
            return_quantum_stats=True,
        )

        # Sequential Hybrid B3: (L-1) KvN layers -> Rollout -> 1 SWA layer
        if is_hybrid and len(self.layers) > 0:
            assert position_ids is not None
            mask_position_ids = get_base_indices_for_anchored_blocks(
                position_ids[0, anchor_positions], self.block_size
            )
            pos_ids_cat = torch.cat(
                [position_ids, mask_position_ids.unsqueeze(0)], dim=1
            )
            position_embeddings = self.rotary_emb(hidden_states, pos_ids_cat)
            swa_query = hidden_blocks.view(1, max_anchors * self.block_size, -1)
            for layer_idx, swa_layer in enumerate(self.layers):
                swa_query = swa_layer(
                    hidden_states=swa_query,
                    target_hidden=fc_output,
                    attention_mask=sliding_window_attn_mask
                    if layer_idx in self.sliding_window_indices
                    else full_attn_mask,
                    position_ids=pos_ids_cat,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_blocks = swa_query.view(max_anchors, self.block_size, -1)

        if self.intrablock_mixer is not None:
            hidden_blocks = self.intrablock_mixer(hidden_blocks)

        mask_tokens_size = max_anchors * self.block_size
        hidden = self.norm(hidden_blocks.view(1, mask_tokens_size, -1))
        logits = self.lm_head(hidden)

        if self.kraus_head is not None:
            kraus_delta = self.kraus_head.forward_topc_logit_delta(
                logits=logits.view(max_anchors, self.block_size, -1),
                q_obs=q_obs,
                p_obs=p_obs,
                lm_head_weight=self.lm_head.weight,
            )
            logits = logits + kraus_delta.view(1, mask_tokens_size, -1)

        # Ensure zero unused DDP parameters even if BornConfidenceHead is disabled in
        # ablations
        if self.born_confidence_head is None and self.config.use_lindblad_purity:
            logits = logits + (0.0 * purity.sum()).to(logits.dtype)

        born_pack = torch.cat([born_obs, p_vac, purity], dim=-1)
        return (
            hidden,
            logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            born_pack,
        )

    def _state_torch_dtype(self) -> torch.dtype:
        return (
            torch.float32
            if self.config.kvn_state_dtype == "float32"
            else torch.bfloat16
        )

    def allocate_state_cache(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> Tensor:
        """Allocate persistent O(1) state buffer [L_kvn, S_vars, B, M] for vLLM."""
        n_vars = self.config.num_recurrent_state_vars()
        return torch.zeros(
            (len(self.kvn_layers), n_vars, batch_size, self.config.kvn_n_modes),
            dtype=self._state_torch_dtype(),
            device=device,
        )

    @torch.no_grad()
    def prefill_kvn_state(
        self,
        input_ids: Tensor,  # [B, T]
        fc_hidden: Tensor,  # [B, T, D] or [B, T, num_aux * D]
        document_ids: Tensor | None = None,  # [B, T]
        initial_state: Tensor | dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Run O(T log T) parallel scan over a prompt/chunk and return persistent
        serving state dict.

        Supports seamless multi-chunk prefill: passing the returned dict back as
        ``initial_state`` on the next chunk preserves exact ``fc_{t-1}`` boundary shift
        and
        FP32 ``(q, p[, nu])`` continuity with zero drift vs single-chunk prefill.
        """
        if not self.kvn_layers:
            raise RuntimeError("prefill_kvn_state requires kvn_layers > 0.")

        bsz, seq_len = input_ids.shape
        dev = input_ids.device
        fc_out = self._resolve_fc_output(fc_hidden)
        tok_emb = self.embed_tokens(input_ids).to(fc_out.dtype)

        prev_last_fc: Tensor | None = None
        prev_last_doc_id: Tensor | None = None
        init_cache_tensor: Tensor | None = None

        if isinstance(initial_state, dict):
            init_cache_tensor = initial_state.get("state_cache")
            prev_last_fc = initial_state.get("last_fc")
            prev_last_doc_id = initial_state.get("last_doc_id")
        elif isinstance(initial_state, Tensor):
            init_cache_tensor = initial_state

        x, reset_mask = self._prepare_causal_kvn_inputs(
            fc_output=fc_out,
            tok_emb=tok_emb,
            document_ids=document_ids,
            prev_last_fc=prev_last_fc,
            prev_last_doc_id=prev_last_doc_id,
        )

        st_dtype = self._state_torch_dtype()
        n_vars = self.config.num_recurrent_state_vars()
        layer_states: list[Tensor] = []
        dtheta_last_list: list[Tensor] = []

        for l_idx, kvn_layer in enumerate(self.kvn_layers):
            l_init: tuple[Tensor, ...] | None = None
            if init_cache_tensor is not None:
                q0 = init_cache_tensor[l_idx, 0]
                p0 = init_cache_tensor[l_idx, 1]
                nu0 = (
                    init_cache_tensor[l_idx, 2]
                    if init_cache_tensor.size(1) >= 3
                    else None
                )
                l_init = (q0, p0, nu0) if nu0 is not None else (q0, p0)

            x, q_seq, p_seq, dtheta_seq, nu_seq = kvn_layer(
                x,
                reset_mask=reset_mask,
                initial_state=l_init,
                state_dtype=st_dtype,
                return_nu=True,
            )
            if seq_len > 0:
                q_end = q_seq[:, -1, :]
                p_end = p_seq[:, -1, :]
                nu_end = nu_seq[:, -1, :]
                dtheta_last_list.append(dtheta_seq[:, -1, :])
            else:
                q_end = (
                    l_init[0].to(st_dtype)
                    if l_init is not None
                    else torch.zeros(
                        (bsz, kvn_layer.n_modes), dtype=st_dtype, device=dev
                    )
                )
                p_end = (
                    l_init[1].to(st_dtype)
                    if l_init is not None
                    else torch.zeros_like(q_end)
                )
                nu_end = (
                    l_init[2].to(st_dtype)
                    if (l_init is not None and len(l_init) >= 3)
                    else torch.zeros_like(q_end)
                )
                dtheta_last_list.append(
                    torch.zeros((bsz, kvn_layer.n_modes), dtype=x.dtype, device=dev)
                )

            if n_vars == 3:
                layer_states.append(torch.stack([q_end, p_end, nu_end], dim=0))
            else:
                layer_states.append(torch.stack([q_end, p_end], dim=0))

        state_cache = torch.stack(layer_states, dim=0)
        last_fc = (
            fc_out[:, -1:, :].detach().clone()
            if seq_len > 0
            else (
                prev_last_fc.clone()
                if prev_last_fc is not None
                else torch.zeros(
                    (bsz, 1, fc_out.size(-1)), dtype=fc_out.dtype, device=dev
                )
            )
        )
        last_doc_id = (
            document_ids[:, -1].detach().clone()
            if (document_ids is not None and seq_len > 0)
            else torch.zeros((bsz,), dtype=torch.long, device=dev)
        )
        last_tokens = (
            input_ids[:, -min(2, seq_len) :].detach().clone()
            if seq_len > 0
            else torch.zeros((bsz, 1), dtype=torch.long, device=dev)
        )
        return {
            "state_cache": state_cache,
            "last_fc": last_fc,
            "last_doc_id": last_doc_id,
            "last_tokens": last_tokens,
            "anchor_h": x[:, -1, :].detach()
            if seq_len > 0
            else torch.zeros_like(last_fc[:, 0]),
            "dtheta_last": torch.stack(dtheta_last_list, dim=0),
        }

    def _step_markov_and_engram_logits(
        self,
        base_logits_k: Tensor,  # [B, V_draft]
        h_k: Tensor,  # [B, D]
        cur_prev_id: Tensor,  # [B]
        cur_prev_prev_id: Tensor | None,  # [B]
        cur_prev3_id: Tensor | None,  # [B]
        rnn_state: Tensor | None,  # [B, R]
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Compute single-horizon (logits_k, p_emb_k, next_rnn_state) with MarkovHead +
        SpeculativeEngram.
        """
        logits_k = base_logits_k
        engram_delta = (
            self.engram(cur_prev_id, cur_prev_prev_id, cur_prev3_id)
            if self.engram is not None
            else None
        )

        p_emb_k: Tensor | None = None
        if self.markov_head is not None:
            p_emb_k = self.markov_head.prev_embeddings(cur_prev_id)
            if engram_delta is not None:
                p_emb_k = p_emb_k + engram_delta.to(p_emb_k.dtype)
            w_dtype = self.markov_head.markov_w2.weight.dtype
            p_emb_cast = p_emb_k.to(w_dtype)
            h_cast = h_k.to(w_dtype)
            if self.markov_head.head_type == "vanilla":
                bias_k = self.markov_head.markov_w2(p_emb_cast)
            elif self.markov_head.head_type == "gated":
                gate = torch.sigmoid(
                    self.markov_head.gate_proj(torch.cat([h_cast, p_emb_cast], dim=-1))
                )
                bias_k = self.markov_head.markov_w2(gate * p_emb_cast)
            else:
                assert rnn_state is not None
                z = torch.cat([rnn_state.to(w_dtype), p_emb_cast, h_cast], dim=-1)
                g_raw, c_raw, o_raw = self.markov_head.joint_proj(z).chunk(3, dim=-1)
                gate = torch.sigmoid(g_raw)
                rnn_state = gate * rnn_state.to(w_dtype) + (1.0 - gate) * torch.tanh(
                    c_raw
                )
                bias_k = self.markov_head.markov_w2(torch.tanh(o_raw))
            logits_k = logits_k + bias_k.to(logits_k.dtype)
        elif engram_delta is not None:
            logits_k = logits_k + F.linear(
                engram_delta.to(self.lm_head.weight.dtype), self.lm_head.weight
            ).to(logits_k.dtype)

        return logits_k, p_emb_k, rnn_state

    def _map_draft_to_target_ids(self, draft_idx: Tensor) -> Tensor:
        if self.use_draft_vocab and self.d2t is not None:
            return draft_idx + self.d2t[draft_idx]
        return draft_idx

    def _decode_draft_block(
        self,
        hidden_blocks: Tensor,  # [B, K, D]
        base_logits: Tensor,  # [B, K, V_draft]
        anchor_token_ids: Tensor,  # [B]
        born_features: Tensor | None = None,
        p_vac: Tensor | None = None,
        purity: Tensor | None = None,
        temperature: float = 0.0,
        prev_context_ids: Tensor | None = None,  # optional [B, <=2] prior tokens
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Autoregressively decode K speculative tokens (greedy or stochastic) + Born
        confidence.
        """
        bsz = hidden_blocks.size(0)
        dev = hidden_blocks.device
        cur_prev_id = anchor_token_ids.long()
        if prev_context_ids is not None and prev_context_ids.size(-1) >= 1:
            cur_prev_prev_id = prev_context_ids[:, -1].long()
            cur_prev3_id = (
                prev_context_ids[:, -2].long()
                if prev_context_ids.size(-1) >= 2
                else torch.zeros_like(cur_prev_id)
            )
        else:
            cur_prev_prev_id = torch.zeros_like(cur_prev_id)
            cur_prev3_id = torch.zeros_like(cur_prev_id)

        rnn_state = (
            hidden_blocks.new_zeros(bsz, self.markov_head.markov_rank)
            if (self.markov_head is not None and self.markov_head.head_type == "rnn")
            else None
        )

        drafted_target_ids: list[Tensor] = []
        draft_probs_list: list[Tensor] = []
        prev_embs: list[Tensor] = []

        for k in range(self.block_size):
            logits_k, p_emb_k, rnn_state = self._step_markov_and_engram_logits(
                base_logits_k=base_logits[:, k, :],
                h_k=hidden_blocks[:, k, :],
                cur_prev_id=cur_prev_id,
                cur_prev_prev_id=cur_prev_prev_id,
                cur_prev3_id=cur_prev3_id,
                rnn_state=rnn_state,
            )
            if p_emb_k is not None:
                prev_embs.append(p_emb_k)

            if temperature > 0.0:
                probs_k = torch.softmax(logits_k.float() / float(temperature), dim=-1)
                draft_idx_k = torch.multinomial(probs_k, num_samples=1).squeeze(-1)
            else:
                probs_k = torch.softmax(logits_k.float(), dim=-1)
                draft_idx_k = logits_k.argmax(dim=-1)

            draft_probs_list.append(probs_k)
            target_id_k = self._map_draft_to_target_ids(draft_idx_k)
            drafted_target_ids.append(target_id_k)

            if self.config.sample_from_anchor or k > 0:
                cur_prev3_id = cur_prev_prev_id
                cur_prev_prev_id = cur_prev_id
                cur_prev_id = target_id_k

        draft_tokens = torch.stack(drafted_target_ids, dim=1)
        draft_probs = torch.stack(draft_probs_list, dim=1)
        prev_emb_block = torch.stack(prev_embs, dim=1) if prev_embs else None

        if self.born_confidence_head is not None:
            conf_logits = self.born_confidence_head(
                hidden_blocks=hidden_blocks,
                prev_emb=prev_emb_block
                if self.config.confidence_head_with_markov
                else None,
                born_features=born_features,
                p_vac=p_vac,
                purity=purity,
            )
            conf_probs = torch.sigmoid(conf_logits.float())
        elif self.confidence_head is not None:
            if self.config.confidence_head_with_markov and prev_emb_block is not None:
                conf_in = torch.cat(
                    [hidden_blocks, prev_emb_block.to(hidden_blocks.dtype)], dim=-1
                )
            else:
                conf_in = hidden_blocks
            conf_probs = torch.sigmoid(self.confidence_head(conf_in).float())
        else:
            conf_probs = torch.ones(
                (bsz, self.block_size), dtype=torch.float32, device=dev
            )
        return draft_tokens, conf_probs, draft_probs

    def _decode_draft_block_greedy(
        self,
        hidden_blocks: Tensor,
        base_logits: Tensor,
        anchor_token_ids: Tensor,
        born_features: Tensor | None = None,
        prev_context_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Backward-compatible greedy block decoder."""
        draft_tokens, conf_probs, _ = self._decode_draft_block(
            hidden_blocks=hidden_blocks,
            base_logits=base_logits,
            anchor_token_ids=anchor_token_ids,
            born_features=born_features,
            temperature=0.0,
            prev_context_ids=prev_context_ids,
        )
        return draft_tokens, conf_probs

    def _advance_layers_and_rollout(
        self,
        prev_fc_block: Tensor,
        token_ids_block: Tensor,
        valid_lens: Tensor,
        state_cache: Tensor | dict[str, Tensor],
        reset_seq_mask: Tensor | None = None,
    ) -> tuple[
        Tensor | dict[str, Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Shared multi-step state advance + Koopman rollout for chain & Kraus tree
        decoding.
        """
        if self.config.kvn_backbone_type != "kvn" or self.rollout is None:
            raise RuntimeError(
                "Stateful Zero-KV proposer requires kvn_backbone_type == 'kvn'."
            )

        is_dict = isinstance(state_cache, dict)
        cache_tensor = state_cache["state_cache"] if is_dict else state_cache
        assert isinstance(cache_tensor, Tensor)

        bsz, s_max = token_ids_block.shape
        dev = token_ids_block.device
        tok_emb = self.embed_tokens(token_ids_block)

        fc_resolved = self._resolve_fc_output(prev_fc_block)
        if is_dict and "last_fc" in state_cache and self.config.kvn_strict_causal_shift:
            # When state_dict tracks unshifted fc_hidden chunks automatically:
            # caller may pass either pre-shifted prev_fc_block OR raw fc_hidden when
            # "unshifted_fc" is flagged.
            x_cur = fc_resolved.to(tok_emb.dtype) + tok_emb
        else:
            x_cur = fc_resolved.to(tok_emb.dtype) + tok_emb

        st_dtype = cache_tensor.dtype
        has_nu_slot = cache_tensor.size(1) >= 3
        v_lens_dev = valid_lens.to(dev)
        last_step = (v_lens_dev - 1).clamp(min=0)

        new_layer_states: list[Tensor] = []
        q_anchors_per_layer: list[Tensor] = []
        p_anchors_per_layer: list[Tensor] = []
        dtheta_anchors_per_layer: list[Tensor] = []
        nu_anchors_per_layer: list[Tensor] = []

        for l_idx, kvn_layer in enumerate(self.kvn_layers):
            q_state = cache_tensor[l_idx, 0]
            p_state = cache_tensor[l_idx, 1]
            nu_state = (
                cache_tensor[l_idx, 2] if has_nu_slot else torch.zeros_like(q_state)
            )

            if reset_seq_mask is not None:
                keep_seq = (~reset_seq_mask.to(dev)).to(q_state.dtype).unsqueeze(-1)
                q_state = q_state * keep_seq
                p_state = p_state * keep_seq
                nu_state = nu_state * keep_seq

            x_cur, q_fin, p_fin, nu_fin, dtheta_seq = kvn_layer.multistep_decode(
                x_block=x_cur,
                q_init=q_state,
                p_init=p_state,
                valid_lens=v_lens_dev,
                nu_init=nu_state if kvn_layer.use_lindblad_purity else None,
                state_dtype=st_dtype,
            )
            nu_saved = nu_fin if nu_fin is not None else torch.zeros_like(q_fin)

            if has_nu_slot:
                new_layer_states.append(torch.stack([q_fin, p_fin, nu_saved], dim=0))
            else:
                new_layer_states.append(torch.stack([q_fin, p_fin], dim=0))

            gather_dt_idx = last_step.view(bsz, 1, 1).expand(bsz, 1, kvn_layer.n_modes)
            dtheta_anchor = dtheta_seq.gather(dim=1, index=gather_dt_idx).squeeze(1)

            q_anchors_per_layer.append(q_fin)
            p_anchors_per_layer.append(p_fin)
            dtheta_anchors_per_layer.append(dtheta_anchor)
            nu_anchors_per_layer.append(nu_saved)

        cache_tensor.copy_(torch.stack(new_layer_states, dim=0))

        gather_h_idx = last_step.view(bsz, 1, 1).expand(bsz, 1, x_cur.size(-1))
        anchor_h = x_cur.gather(dim=1, index=gather_h_idx).squeeze(1)
        anchor_token_ids = token_ids_block.gather(
            dim=1, index=last_step.view(bsz, 1)
        ).squeeze(1)

        anchor_q, anchor_p, anchor_dtheta, anchor_nu = (
            self.rollout.aggregate_multilayer_states(
                q_anchors_per_layer,
                p_anchors_per_layer,
                dtheta_anchors_per_layer,
                nu_anchors_per_layer,
            )
        )
        raw_blocks, born_obs, p_vac, purity, q_obs, p_obs = self.rollout(
            anchor_h=anchor_h,
            anchor_q=anchor_q,
            anchor_p=anchor_p,
            anchor_dtheta=anchor_dtheta,
            anchor_nu=anchor_nu,
            return_quantum_stats=True,
        )
        if self.intrablock_mixer is not None:
            raw_blocks = self.intrablock_mixer(raw_blocks)

        hidden_blocks = self.norm(raw_blocks)
        base_logits = self.lm_head(hidden_blocks)
        if self.kraus_head is not None:
            base_logits = base_logits + self.kraus_head.forward_topc_logit_delta(
                logits=base_logits,
                q_obs=q_obs,
                p_obs=p_obs,
                lm_head_weight=self.lm_head.weight,
            )

        updated_cache: Tensor | dict[str, Tensor] = state_cache
        if is_dict:
            updated_cache = dict(state_cache)
            updated_cache["state_cache"] = cache_tensor

        return (
            updated_cache,
            hidden_blocks,
            base_logits,
            anchor_token_ids,
            born_obs,
            p_vac,
            purity,
            q_obs,
            p_obs,
        )

    @torch.no_grad()
    def advance_and_draft_step(
        self,
        prev_fc_block: Tensor,
        token_ids_block: Tensor,
        valid_lens: Tensor,
        state_cache: Tensor | dict[str, Tensor],
        reset_seq_mask: Tensor | None = None,
        temperature: float = 0.0,
        return_details: bool = False,
        prev_context_ids: Tensor | None = None,
    ):
        """Static-shape CUDA-Graph-compatible vLLM proposer step."""
        (
            updated_cache,
            hidden_blocks,
            base_logits,
            anchor_token_ids,
            born_obs,
            p_vac,
            purity,
            _,
            _,
        ) = self._advance_layers_and_rollout(
            prev_fc_block=prev_fc_block,
            token_ids_block=token_ids_block,
            valid_lens=valid_lens,
            state_cache=state_cache,
            reset_seq_mask=reset_seq_mask,
        )

        draft_tokens, conf_probs, draft_probs = self._decode_draft_block(
            hidden_blocks=hidden_blocks,
            base_logits=base_logits,
            anchor_token_ids=anchor_token_ids,
            born_features=born_obs,
            p_vac=p_vac,
            purity=purity,
            temperature=temperature,
            prev_context_ids=prev_context_ids,
        )
        if not return_details:
            return draft_tokens, conf_probs

        prefix_survival = torch.cumprod(conf_probs, dim=-1)
        draft_entropy = -(draft_probs * draft_probs.clamp_min(1e-12).log()).sum(dim=-1)
        details = {
            "draft_probs": draft_probs,
            "draft_entropy": draft_entropy,
            "confidences": conf_probs,
            "prefix_survival": prefix_survival,
            "purity": purity.float().mean(dim=-1),
            "p_vac": p_vac.float().mean(dim=-1),
            "mode_purity": purity.float(),
            "dyad_p_vac": p_vac.float(),
        }
        return draft_tokens, updated_cache, details

    @torch.no_grad()
    def draft_kraus_tree_step(
        self,
        confirmed_fc_hidden: Tensor,
        confirmed_token_ids: Tensor,
        valid_lens: Tensor,
        state_cache: Tensor | dict[str, Tensor],
        tree_budget: int = 24,
        branch_width: int = 3,
        temperature: float = 0.0,
        reset_seq_mask: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], Tensor | dict[str, Tensor]]:
        """Expand a SpecInfer-compatible speculative tree via diagonal Kraus POVM
        back-action (Axe 5).

        At each parent node u, evaluates active top-C diagonal Kraus operators
        satisfying
        ``sum_{y in C_k} M_y^dagger M_y = I_{M+1}`` identically, selects/samples up to
        ``branch_width`` distinct sibling tokens without replacement (recording the
        exact
        sibling-conditioned proposal distribution ``tree_q_probs`` for lossless
        SpecInfer
        verification), and collapses child modal amplitudes ``(q_u, p_u)`` via Lüders
        rule.
        """
        if tree_budget <= 0:
            raise ValueError(f"tree_budget must be > 0, got {tree_budget}.")

        (
            updated_cache,
            hidden_blocks,
            base_logits,
            anchor_token_ids,
            born_obs,
            p_vac,
            purity,
            q_obs,
            p_obs,
        ) = self._advance_layers_and_rollout(
            prev_fc_block=confirmed_fc_hidden,
            token_ids_block=confirmed_token_ids,
            valid_lens=valid_lens,
            state_cache=state_cache,
            reset_seq_mask=reset_seq_mask,
        )

        bsz, k_horizons, v_draft = base_logits.shape
        dev = base_logits.device
        eff_w = max(1, min(branch_width, v_draft))

        # Determine node allocation per depth so total expanded nodes == tree_budget
        # If geometric tree sum_{k=0}^{K-1} eff_w^{k+1} < tree_budget, widen root/early
        # branching
        max_geom = sum(eff_w ** (k + 1) for k in range(k_horizons))
        root_w = eff_w
        if max_geom < tree_budget and v_draft > eff_w:
            root_w = min(v_draft, max(eff_w, tree_budget))

        tree_draft_ids = torch.zeros((bsz, tree_budget), dtype=torch.long, device=dev)
        tree_parent_idx = torch.full(
            (bsz, tree_budget), -1, dtype=torch.long, device=dev
        )
        tree_depths = torch.zeros((bsz, tree_budget), dtype=torch.long, device=dev)
        tree_q_probs = torch.zeros(
            (bsz, tree_budget, v_draft), dtype=torch.float32, device=dev
        )
        tree_token_probs = torch.zeros(
            (bsz, tree_budget), dtype=torch.float32, device=dev
        )
        tree_survival = torch.zeros((bsz, tree_budget), dtype=torch.float32, device=dev)
        max_povm_err = torch.zeros((), dtype=torch.float32, device=dev)

        # Per-batch expansion (bsz in speculative serving is small; keeps
        # survival-ranked beam exact per req)
        for b in range(bsz):
            # Frontier entries: (node_idx, depth_next, prev_id, prev_prev_id, prev3_id,
            # rnn_st, h_back, survival)
            init_rnn = (
                hidden_blocks.new_zeros((1, self.markov_head.markov_rank))
                if (
                    self.markov_head is not None and self.markov_head.head_type == "rnn"
                )
                else None
            )
            zero_id = torch.zeros((1,), dtype=torch.long, device=dev)
            zero_back = torch.zeros(
                (1, hidden_blocks.size(-1)),
                dtype=hidden_blocks.dtype,
                device=dev,
            )

            frontier: list[
                tuple[int, int, Tensor, Tensor, Tensor, Tensor | None, Tensor, float]
            ] = [
                (
                    -1,
                    0,
                    anchor_token_ids[b : b + 1].long(),
                    zero_id,
                    zero_id,
                    init_rnn,
                    zero_back,
                    1.0,
                )
            ]
            written = 0

            for depth in range(k_horizons):
                if written >= tree_budget or not frontier:
                    break

                # Sort frontier at current depth by descending cumulative survival
                frontier.sort(key=lambda item: item[7], reverse=True)
                next_frontier: list[
                    tuple[
                        int, int, Tensor, Tensor, Tensor, Tensor | None, Tensor, float
                    ]
                ] = []

                h_base_d = hidden_blocks[b : b + 1, depth, :]
                logits_base_d = base_logits[b : b + 1, depth, :]
                q_base_d = q_obs[b : b + 1, depth, :]
                p_base_d = p_obs[b : b + 1, depth, :]
                born_d = born_obs[b : b + 1, depth : depth + 1, :]
                pvac_d = p_vac[b : b + 1, depth : depth + 1, :]
                pur_d = purity[b : b + 1, depth : depth + 1, :]

                width_here = root_w if depth == 0 else eff_w
                reserve_spine = (
                    max(0, k_horizons - depth - 1) if tree_budget >= k_horizons else 0
                )
                level_limit = max(written + 1, tree_budget - reserve_spine)
                level_limit = min(level_limit, tree_budget)

                for (
                    parent_node,
                    _,
                    prev_id,
                    prev_prev_id,
                    prev3_id,
                    rnn_st,
                    h_back,
                    parent_surv,
                ) in frontier:
                    if written >= level_limit:
                        break

                    n_children = min(width_here, level_limit - written, v_draft)
                    h_node = h_base_d + h_back
                    if parent_node >= 0:
                        # Add back-action logit shift from parent's Kraus measurement
                        # collapse
                        logits_in = logits_base_d + F.linear(
                            h_back.to(self.lm_head.weight.dtype), self.lm_head.weight
                        ).to(logits_base_d.dtype)
                    else:
                        logits_in = logits_base_d

                    logits_u, p_emb_u, next_rnn_u = self._step_markov_and_engram_logits(
                        base_logits_k=logits_in,
                        h_k=h_node,
                        cur_prev_id=prev_id,
                        cur_prev_prev_id=prev_prev_id,
                        cur_prev3_id=prev3_id,
                        rnn_state=rnn_st,
                    )

                    # Evaluate diagonal Kraus POVM over active top-C candidates
                    povm_diag_u: Tensor | None = None
                    top_idx_u: Tensor | None = None
                    if self.kraus_head is not None:
                        eff_c = min(
                            self.kraus_head.top_c,
                            v_draft,
                            max(n_children, self.kraus_head.top_c),
                        )
                        eff_c = min(eff_c, v_draft)
                        top_idx_u = logits_u.topk(eff_c, dim=-1).indices  # [1, C]
                        cand_logits = logits_u.gather(dim=-1, index=top_idx_u)
                        cand_vecs = self.lm_head.weight.index_select(
                            0, top_idx_u.view(-1)
                        ).unsqueeze(0)
                        _, povm_diag_u, _ = self.kraus_head.compute_povm(
                            cand_logits, cand_vecs, q_base_d, p_base_d
                        )
                        comp_err = (povm_diag_u.sum(dim=-2) - 1.0).abs().max()
                        max_povm_err = torch.maximum(max_povm_err, comp_err)

                    # Node acceptance confidence
                    if self.born_confidence_head is not None:
                        conf_u = torch.sigmoid(
                            self.born_confidence_head(
                                hidden_blocks=h_node.unsqueeze(1),
                                prev_emb=p_emb_u.unsqueeze(1)
                                if (
                                    p_emb_u is not None
                                    and self.config.confidence_head_with_markov
                                )
                                else None,
                                born_features=born_d,
                                p_vac=pvac_d,
                                purity=pur_d,
                            ).float()
                        ).item()
                    else:
                        conf_u = 1.0

                    if temperature > 0.0:
                        base_q = torch.softmax(
                            logits_u[0].float() / float(temperature), dim=-1
                        )
                    else:
                        base_q = torch.softmax(logits_u[0].float(), dim=-1)
                        greedy_top = logits_u[0].topk(n_children, dim=-1).indices

                    avail_mask = torch.ones_like(base_q)
                    for r in range(n_children):
                        masked_q = base_q * avail_mask
                        mass = masked_q.sum()
                        if float(mass.item()) <= 1e-12:
                            q_r = avail_mask / avail_mask.sum().clamp_min(1.0)
                        else:
                            q_r = masked_q / mass

                        if temperature > 0.0:
                            draft_tok = int(
                                torch.multinomial(q_r, num_samples=1).item()
                            )
                        else:
                            draft_tok = int(greedy_top[r].item())

                        avail_mask[draft_tok] = 0.0
                        draft_tok_t = torch.tensor(
                            [draft_tok], dtype=torch.long, device=dev
                        )
                        target_tok_t = self._map_draft_to_target_ids(draft_tok_t)
                        tok_prob = float(q_r[draft_tok].item())
                        child_surv = float(parent_surv * conf_u * max(tok_prob, 1e-4))

                        node_idx = written
                        written += 1
                        tree_draft_ids[b, node_idx] = target_tok_t[0]
                        tree_parent_idx[b, node_idx] = parent_node
                        tree_depths[b, node_idx] = depth + 1
                        tree_q_probs[b, node_idx] = q_r
                        tree_token_probs[b, node_idx] = tok_prob
                        tree_survival[b, node_idx] = child_surv

                        # Compute Lüders-Kraus collapsed back-action for child depth + 1
                        if (
                            self.kraus_head is not None
                            and povm_diag_u is not None
                            and top_idx_u is not None
                            and (depth + 1) < k_horizons
                        ):
                            match_pos = (top_idx_u[0] == draft_tok).nonzero(
                                as_tuple=False
                            )
                            if match_pos.numel() > 0:
                                c_pos = int(match_pos[0, 0].item())
                                povm_row = povm_diag_u[:, c_pos, :]
                            else:
                                povm_row = torch.full(
                                    (1, self.kraus_head.n_modes + 1),
                                    1.0 / float(max(1, eff_c)),
                                    dtype=torch.float32,
                                    device=dev,
                                )
                            _, _, raw_h_back = self.kraus_head.collapse_branch_state(
                                q_obs[b : b + 1, depth + 1, :],
                                p_obs[b : b + 1, depth + 1, :],
                                povm_row,
                                hidden_blocks.dtype,
                            )
                            _, _, top1_h_back = self.kraus_head.collapse_branch_state(
                                q_obs[b : b + 1, depth + 1, :],
                                p_obs[b : b + 1, depth + 1, :],
                                povm_diag_u[:, 0, :],
                                hidden_blocks.dtype,
                            )
                            child_h_back = raw_h_back - top1_h_back
                        else:
                            child_h_back = zero_back

                        if (depth + 1) < k_horizons:
                            next_prev_id = (
                                target_tok_t
                                if self.config.sample_from_anchor
                                else (prev_id if depth == 0 else target_tok_t)
                            )
                            next_frontier.append(
                                (
                                    node_idx,
                                    depth + 1,
                                    next_prev_id,
                                    prev_id,
                                    prev_prev_id,
                                    next_rnn_u.clone()
                                    if next_rnn_u is not None
                                    else None,
                                    child_h_back,
                                    child_surv,
                                )
                            )

                frontier = next_frontier

        # Construct causal tree ancestor attention mask [B, tree_budget, tree_budget]
        tree_attn_mask = (
            torch.eye(tree_budget, dtype=torch.bool, device=dev)
            .unsqueeze(0)
            .expand(bsz, tree_budget, tree_budget)
            .clone()
        )
        for b in range(bsz):
            for i in range(tree_budget):
                p_idx = int(tree_parent_idx[b, i].item())
                if 0 <= p_idx < i:
                    tree_attn_mask[b, i] = (
                        tree_attn_mask[b, i] | tree_attn_mask[b, p_idx]
                    )

        tree_dict = {
            "tree_draft_ids": tree_draft_ids,
            "tree_parent_idx": tree_parent_idx,
            "tree_depths": tree_depths,
            "tree_attn_mask": tree_attn_mask,
            "tree_q_probs": tree_q_probs,
            "tree_token_probs": tree_token_probs,
            "tree_survival": tree_survival,
            "purity": purity.float().mean(dim=(1, 2)),
            "p_vac": p_vac.float().mean(dim=(1, 2)),
            "povm_completeness_max_err": max_povm_err,
        }
        return tree_dict, updated_cache

    @torch.no_grad()
    def _rollout_self_cond_prev_tokens(
        self,
        base_logits: Tensor,  # [N, K, V_draft]
        hidden_blocks: Tensor,  # [N, K, D]
        teacher_prev_ids: Tensor,  # [N, K]
    ) -> Tensor:
        """Stop-grad 1st-pass Markov/Engram rollout y_hat_{k-1} ~ q_{k-1} (Axe 2)."""
        num_blocks, block, _ = base_logits.shape
        if block <= 1:
            return teacher_prev_ids

        cur_prev_id = teacher_prev_ids[:, 0].long()
        cur_prev_prev_id = torch.zeros_like(cur_prev_id)
        cur_prev3_id = torch.zeros_like(cur_prev_id)
        rnn_state = (
            hidden_blocks.new_zeros(num_blocks, self.markov_head.markov_rank)
            if (self.markov_head is not None and self.markov_head.head_type == "rnn")
            else None
        )

        sampled_prev: list[Tensor] = [cur_prev_id]
        for k in range(block - 1):
            logits_k, _, rnn_state = self._step_markov_and_engram_logits(
                base_logits_k=base_logits[:, k, :],
                h_k=hidden_blocks[:, k, :],
                cur_prev_id=cur_prev_id,
                cur_prev_prev_id=cur_prev_prev_id,
                cur_prev3_id=cur_prev3_id,
                rnn_state=rnn_state,
            )
            probs_k = torch.softmax(logits_k.float(), dim=-1)
            draft_idx_k = torch.multinomial(probs_k, num_samples=1).squeeze(-1)
            target_id_k = self._map_draft_to_target_ids(draft_idx_k).clamp(
                min=0, max=self.verifier_vocab_size - 1
            )
            if self.config.sample_from_anchor or k > 0:
                cur_prev3_id = cur_prev_prev_id
                cur_prev_prev_id = cur_prev_id
                cur_prev_id = target_id_k
            sampled_prev.append(cur_prev_id)

        sc_prev = torch.stack(sampled_prev, dim=1)
        sc_mask = torch.rand((num_blocks, 1), device=base_logits.device) < float(
            self.config.self_cond_rollout_prob
        )
        return torch.where(sc_mask, sc_prev, teacher_prev_ids)

    @_kvn_compile
    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        loss_mask: Tensor,
        verifier_last_hidden_states: Tensor,
        document_ids: Tensor,
        position_ids: Tensor | None = None,
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[..., Tensor] = _DEFAULT_TV_FN,
        loss_implementation: str = "fused",
        gamma: float = 4.0,
        max_anchors: int = 512,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        kvn_lasso_weight: float | None = None,
        **kwargs,
    ):
        (
            hidden,
            logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            born_features,
        ) = self._kvn_backbone_forward(
            hidden_states=hidden_states,
            input_ids=input_ids,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last_hidden_states,
            document_ids=document_ids,
            position_ids=position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )

        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            prev_token_ids = block_tokens
        else:
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )
        hidden_blocks = hidden.view(num_blocks, block, -1)
        logits_blocks = logits.view(num_blocks, block, -1)

        if self.training and self.config.self_cond_rollout_prob > 0.0:
            prev_token_ids = self._rollout_self_cond_prev_tokens(
                base_logits=logits_blocks.detach(),
                hidden_blocks=hidden_blocks.detach(),
                teacher_prev_ids=prev_token_ids,
            )

        engram_delta = self.engram(prev_token_ids) if self.engram is not None else None

        confidence_logits = None
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            if engram_delta is not None:
                prev_emb = prev_emb + engram_delta.to(prev_emb.dtype)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits_blocks + markov_bias).view(1, mask_tokens_size, -1)
        elif engram_delta is not None:
            engram_logits = F.linear(
                engram_delta.to(self.lm_head.weight.dtype), self.lm_head.weight
            ).to(logits.dtype)
            logits = (logits_blocks + engram_logits).view(1, mask_tokens_size, -1)

        if self.born_confidence_head is not None:
            confidence_logits = self.born_confidence_head(
                hidden_blocks=hidden_blocks,
                prev_emb=prev_emb if self.config.confidence_head_with_markov else None,
                born_features=born_features,
            ).reshape(1, mask_tokens_size)
        elif self.confidence_head is not None:
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)],
                    dim=-1,
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        target_lse_full, in_vocab_mass, target_argmax_in_vocab = getattr(
            self, "_last_shortlist_stats", (None, None, None)
        )
        loss, metrics = compute_kvn_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            target_lse_full=target_lse_full,
            in_vocab_mass=in_vocab_mass,
            target_argmax_in_vocab=target_argmax_in_vocab,
            loss_implementation=loss_implementation,
        )
        if self.training:
            lasso_pen = self.total_lasso_penalty(lasso_weight=kvn_lasso_weight)
            zero_touch = 0.0 * logits.float().sum()
            if confidence_logits is not None:
                zero_touch = zero_touch + 0.0 * confidence_logits.float().sum()
            loss = loss + lasso_pen.to(loss.dtype) + zero_touch.to(loss.dtype)
        else:
            lasso_pen = torch.zeros((), device=loss.device, dtype=torch.float32)
        metrics["lasso_loss_sum"] = lasso_pen.detach().to(torch.float32)
        metrics["lasso_loss_total"] = torch.ones((), device=loss.device)
        return None, loss, metrics

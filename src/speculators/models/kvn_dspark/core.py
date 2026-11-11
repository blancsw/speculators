from __future__ import annotations

from collections.abc import Callable
import importlib.util
from typing import ClassVar

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig

from speculators.losses import (
    LossConfig,
    eager as eager_losses,
    kl_div_loss,
    resolve_loss_config,
    tv_loss,
)
from speculators.model import SpeculatorModel
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.kvn_dspark.config import KvNDSparkSpeculatorConfig
from speculators.models.kvn_dspark.model_definitions import (
    BornConfidenceHead,
    CausalKvNRFFLayer,
    IntraBlockCausalMixer,
    MultiHorizonKoopmanRollout,
)
from speculators.models.utils import conditional_torch_compile

_HAS_TRITON = importlib.util.find_spec("triton") is not None


def _default_kl_fn(logits: Tensor, targets: Tensor) -> Tensor:
    if _HAS_TRITON and logits.is_cuda:
        return kl_div_loss(logits, targets)
    return eager_losses.kl_div_loss(logits, targets)


def _default_tv_fn(logits: Tensor, targets: Tensor) -> Tensor:
    if _HAS_TRITON and logits.is_cuda:
        return tv_loss(logits, targets)
    return eager_losses.tv_loss(logits, targets)


_DEFAULT_KL_FN = _default_kl_fn
_DEFAULT_TV_FN = _default_tv_fn
_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (_DEFAULT_KL_FN, 1.0)}

__all__ = [
    "KvNDSparkDraftModel",
]


@SpeculatorModel.register("kvn_dspark")
class KvNDSparkDraftModel(DSparkDraftModel):
    """Zero-Drafter-KV-Cache Dequantized Damped Bosonic SSM Speculator Model (Phase 2).

    Supports three controlled scientific variants via ``config.kvn_backbone_type``:
    - ``"kvn"``: 100% Zero-KV-Cache causal damped Bosonic KvN-RFF layers +
      multi-layer Multi-Horizon Koopman Rollout + Zero-KV IntraBlockCausalMixer +
      MarkovHead + interferometric vacuum-normalized BornConfidenceHead.
    - ``"hybrid_1swa_2kvn"``: 1 SWA-128 attention layer + (L-1) KvN-RFF layers.
    - ``"swa"``: Standard DSpark SWA attention stack (baseline).
    """

    config_class: ClassVar[type[KvNDSparkSpeculatorConfig]] = KvNDSparkSpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = [
        "Qwen3DFlashDecoderLayer",
        "CausalKvNRFFLayer",
        "IntraBlockCausalMixer",
    ]

    def _build_kvn_layer(self, hidden_size: int) -> CausalKvNRFFLayer:
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
        )

    def __init__(self, config: KvNDSparkSpeculatorConfig) -> None:
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
            for _ in range(num_draft_layers):
                self.kvn_layers.append(self._build_kvn_layer(hidden_size))
            self.rollout = MultiHorizonKoopmanRollout(
                d_model=hidden_size,
                n_modes=config.kvn_n_modes,
                block_size=self.block_size,
                anchor_rank=config.kvn_anchor_rank,
                enable_interferometer=config.kvn_enable_interferometer,
                vacuum_born_norm=config.kvn_vacuum_born_norm,
                num_layers=len(self.kvn_layers),
                multilayer_rollout=config.kvn_multilayer_rollout,
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
            for _ in range(num_kvn):
                self.kvn_layers.append(self._build_kvn_layer(hidden_size))
            self.rollout = MultiHorizonKoopmanRollout(
                d_model=hidden_size,
                n_modes=config.kvn_n_modes,
                block_size=self.block_size,
                anchor_rank=config.kvn_anchor_rank,
                enable_interferometer=config.kvn_enable_interferometer,
                vacuum_born_norm=config.kvn_vacuum_born_norm,
                num_layers=len(self.kvn_layers),
                multilayer_rollout=config.kvn_multilayer_rollout,
            )
            if config.kvn_intrablock_mixer:
                self.intrablock_mixer = IntraBlockCausalMixer(
                    d_model=hidden_size,
                    block_size=self.block_size,
                    n_heads=min(8, n_heads),
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
            )
            self.confidence_head = None

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: Tensor | None = None,
        d2t: Tensor | None = None,
        **kwargs,
    ) -> KvNDSparkDraftModel:
        """Create a KvNDSparkDraftModel from training CLI/YAML arguments."""
        def _bool_or_default(key: str, default: bool) -> bool:
            val = kwargs.get(key)
            return default if val is None else bool(val)

        config = KvNDSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("kvn_dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=_bool_or_default("enable_confidence_head", True),
            confidence_head_with_markov=_bool_or_default(
                "confidence_head_with_markov", True
            ),
            kvn_backbone_type=kwargs.get("kvn_backbone_type", "kvn"),
            kvn_n_modes=kwargs.get("kvn_n_modes", 256),
            kvn_reupload_layers=kwargs.get("kvn_reupload_layers", 2),
            kvn_reupload_dim=kwargs.get("kvn_reupload_dim", 512),
            kvn_learnable_rff=_bool_or_default("kvn_learnable_rff", True),
            kvn_sigma_g=kwargs.get("kvn_sigma_g", 1.0),
            kvn_lasso_weight=kwargs.get("kvn_lasso_weight", 1e-4),
            kvn_use_born_confidence=_bool_or_default("kvn_use_born_confidence", True),
            kvn_enable_interferometer=_bool_or_default(
                "kvn_enable_interferometer", True
            ),
            kvn_vacuum_born_norm=_bool_or_default("kvn_vacuum_born_norm", True),
            kvn_anchor_rank=kwargs.get("kvn_anchor_rank", 512),
            kvn_multilayer_rollout=_bool_or_default("kvn_multilayer_rollout", True),
            kvn_intrablock_mixer=_bool_or_default("kvn_intrablock_mixer", True),
            kvn_strict_causal_shift=_bool_or_default("kvn_strict_causal_shift", True),
            kvn_ffn_mult=kwargs.get("kvn_ffn_mult", 1.5),
            kvn_state_dtype=kwargs.get("kvn_state_dtype", "bfloat16"),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve KvN-DSpark's compound loss, falling back to eager when Triton is absent."""
        implementation = kwargs.get("loss_implementation", "fused")
        if implementation == "fused" and not _HAS_TRITON:
            implementation = "eager"
        loss_config = resolve_loss_config(kwargs["loss_fn"], implementation)
        tv_loss_fn = resolve_loss_config("tv", implementation)["tv"][0]
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
    def prune_kvn_modes(self, target_modes: int) -> None:
        """Structurally prune all CausalKvNRFFLayer blocks (M_train -> M_infer) and align rollout/confidence heads."""
        if not self.kvn_layers:
            return

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
            # Jointly align mode indices across all layers so softmax(layer_mode_logits, dim=0)
            # and aggregated multi-layer states remain 100% invariant under zero-shot pruning.
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

        if last_top_idx is not None and self.rollout is not None:
            self.rollout.prune_to_indices(last_top_idx)
            if self.born_confidence_head is not None:
                self.born_confidence_head.prune_born_modes(last_top_idx)
        self.config.kvn_n_modes = target_modes
        self.config.kvn_lasso_weight = 0.0

    def _prepare_causal_kvn_inputs(
        self,
        fc_output: Tensor,
        tok_emb: Tensor,
        document_ids: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Construct strictly causal sequence inputs x_t = fc_shifted_t + tok_emb_t and doc reset mask.

        At step t, fc_shifted[:, t] = fc_output[:, t - 1], zeroed at t = 0 and across any
        document boundary (document_ids[:, t] != document_ids[:, t - 1]). Consequently,
        anchor position a strictly observes fc_output[:, 0..a-1] and input_ids[:, 0..a],
        matching DSpark's `kv_base_pos < q_anchor` information set with zero anchor leak
        and zero bonus-token rollback in vLLM.
        """
        bsz, seq_len, _ = fc_output.shape
        reset_mask = torch.zeros(
            (bsz, seq_len), dtype=torch.bool, device=fc_output.device
        )
        reset_mask[:, 0] = True
        if document_ids is not None and seq_len > 1:
            reset_mask[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]

        if not self.config.kvn_strict_causal_shift:
            return fc_output + tok_emb, reset_mask

        fc_shifted = torch.zeros_like(fc_output)
        if seq_len > 1:
            fc_shifted[:, 1:, :] = fc_output[:, :-1, :]
            fc_shifted = fc_shifted.masked_fill(reset_mask.unsqueeze(-1), 0.0)
        return fc_shifted + tok_emb, reset_mask

    @torch.compiler.disable
    def _select_kvn_anchors(
        self, loss_mask: Tensor, max_anchors: int
    ) -> tuple[Tensor, Tensor]:
        """Select valid anchors outside torch.compile to avoid data-dependent nonzero shapes."""
        return select_anchors(loss_mask, max_anchors, self.block_size)

    def _compute_targets_and_mask(
        self,
        verifier_last_hidden_states: Tensor,
        loss_mask: Tensor,
        document_ids: Tensor,
        anchor_positions: Tensor,
        anchor_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states[:, target_indices])
                )
            else:
                verifier_logits = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                if not self.config.sample_from_anchor:
                    verifier_logits = torch.roll(verifier_logits, 1, dims=1)
                targets = verifier_logits[:, anchored_block_indices]

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        same_doc_mask = (
            document_ids[:, anchored_block_indices]
            == document_ids[:, anchor_positions].repeat_interleave(
                self.block_size, dim=1
            )
        )
        aligned_loss_mask = (
            aligned_loss_mask
            * same_doc_mask.to(aligned_loss_mask.dtype)
            * (
                anchor_valid.repeat_interleave(self.block_size)
                .unsqueeze(0)
                .to(aligned_loss_mask.dtype)
            )
        )
        if not self.config.sample_from_anchor:
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
        """Run Pure-KvN, Hybrid, or SWA backbone and return (hidden, logits, targets, mask, indices, born_features)."""
        if self.config.kvn_backbone_type == "swa":
            hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
                super()._backbone_forward(
                    hidden_states,
                    input_ids,
                    loss_mask,
                    verifier_last_hidden_states,
                    document_ids,
                    position_ids,
                    max_anchors=max_anchors,
                    **kwargs,
                )
            )
            return (
                hidden,
                logits,
                targets,
                aligned_loss_mask,
                anchored_block_indices,
                None,
            )

        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if self.config.kvn_backbone_type == "hybrid_1swa_2kvn":
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
            swa_blocks = noise_embedding.view(max_anchors, self.block_size, -1)
        else:
            anchor_positions, anchor_valid = self._select_kvn_anchors(
                loss_mask, max_anchors
            )
            fc_output = self.hidden_norm(self.fc(hidden_states))
            swa_blocks = None

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
        for kvn_layer in self.kvn_layers:
            x, q_seq, p_seq, dtheta_seq = kvn_layer(x, reset_mask=reset_mask)
            q_anchors_per_layer.append(q_seq[0, anchor_positions])
            p_anchors_per_layer.append(p_seq[0, anchor_positions])
            dtheta_anchors_per_layer.append(dtheta_seq[0, anchor_positions])

        assert self.rollout is not None and len(q_anchors_per_layer) > 0
        anchor_q, anchor_p, anchor_dtheta = self.rollout.aggregate_multilayer_states(
            q_anchors_per_layer, p_anchors_per_layer, dtheta_anchors_per_layer
        )
        anchor_h = x[0, anchor_positions]
        hidden_blocks, born_features = self.rollout(
            anchor_h=anchor_h,
            anchor_q=anchor_q,
            anchor_p=anchor_p,
            anchor_dtheta=anchor_dtheta,
        )
        if self.intrablock_mixer is not None:
            hidden_blocks = self.intrablock_mixer(hidden_blocks)
        if swa_blocks is not None:
            hidden_blocks = hidden_blocks + swa_blocks

        mask_tokens_size = max_anchors * self.block_size
        hidden = self.norm(hidden_blocks.view(1, mask_tokens_size, -1))
        logits = self.lm_head(hidden)

        return (
            hidden,
            logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            born_features,
        )

    def allocate_state_cache(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> Tensor:
        """Allocate persistent O(1) recurrent state buffer [L_kvn, 2, B, M] for vLLM."""
        dtype = (
            torch.float32
            if self.config.kvn_state_dtype == "float32"
            else torch.bfloat16
        )
        return torch.zeros(
            (len(self.kvn_layers), 2, batch_size, self.config.kvn_n_modes),
            dtype=dtype,
            device=device,
        )

    def _decode_draft_block_greedy(
        self,
        hidden_blocks: Tensor,  # [B, K, D] post-final-norm hidden states
        base_logits: Tensor,  # [B, K, V_draft] lm_head(hidden_blocks)
        anchor_token_ids: Tensor,  # [B] target vocab token IDs at each anchor
        born_features: Tensor | None = None,  # [B, K, M]
    ) -> tuple[Tensor, Tensor]:
        """Autoregressively apply MarkovHead across K horizons and compute Born confidence."""
        bsz = hidden_blocks.size(0)
        cur_prev_id = anchor_token_ids.long()
        rnn_state = (
            hidden_blocks.new_zeros(bsz, self.markov_head.markov_rank)
            if (self.markov_head is not None and self.markov_head.head_type == "rnn")
            else None
        )

        drafted_target_ids: list[Tensor] = []
        prev_embs: list[Tensor] = []

        for k in range(self.block_size):
            logits_k = base_logits[:, k, :]
            if self.markov_head is not None:
                p_emb_k = self.markov_head.prev_embeddings(cur_prev_id)
                prev_embs.append(p_emb_k)
                w_dtype = self.markov_head.markov_w2.weight.dtype
                p_emb_cast = p_emb_k.to(w_dtype)
                h_k = hidden_blocks[:, k, :].to(w_dtype)
                if self.markov_head.head_type == "vanilla":
                    bias_k = self.markov_head.markov_w2(p_emb_cast)
                elif self.markov_head.head_type == "gated":
                    gate = torch.sigmoid(
                        self.markov_head.gate_proj(
                            torch.cat([h_k, p_emb_cast], dim=-1)
                        )
                    )
                    bias_k = self.markov_head.markov_w2(gate * p_emb_cast)
                else:
                    assert rnn_state is not None
                    z = torch.cat([rnn_state.to(w_dtype), p_emb_cast, h_k], dim=-1)
                    g_raw, c_raw, o_raw = self.markov_head.joint_proj(z).chunk(
                        3, dim=-1
                    )
                    gate = torch.sigmoid(g_raw)
                    rnn_state = gate * rnn_state.to(w_dtype) + (
                        1.0 - gate
                    ) * torch.tanh(c_raw)
                    bias_k = self.markov_head.markov_w2(torch.tanh(o_raw))
                logits_k = logits_k + bias_k.to(logits_k.dtype)

            draft_idx_k = logits_k.argmax(dim=-1)
            if self.use_draft_vocab and self.d2t is not None:
                target_id_k = draft_idx_k + self.d2t[draft_idx_k]
            else:
                target_id_k = draft_idx_k
            drafted_target_ids.append(target_id_k)

            if self.config.sample_from_anchor:
                cur_prev_id = target_id_k
            else:
                cur_prev_id = anchor_token_ids.long() if k == 0 else target_id_k

        draft_tokens = torch.stack(drafted_target_ids, dim=1)
        prev_emb_block = torch.stack(prev_embs, dim=1) if prev_embs else None

        if self.born_confidence_head is not None:
            conf_logits = self.born_confidence_head(
                hidden_blocks=hidden_blocks,
                prev_emb=prev_emb_block
                if self.config.confidence_head_with_markov
                else None,
                born_features=born_features,
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
                (bsz, self.block_size),
                dtype=torch.float32,
                device=hidden_blocks.device,
            )
        return draft_tokens, conf_probs

    @torch.no_grad()
    def advance_and_draft_step(
        self,
        prev_fc_block: Tensor,  # [B, S_max, D] static-shape shifted verifier fc states
        token_ids_block: Tensor,  # [B, S_max] newly accepted tokens + bonus anchor token
        valid_lens: Tensor,  # [B] int64 in [1, S_max]
        state_cache: Tensor,  # [L_kvn, 2, B, M] persistent O(1) recurrent state buffer
        reset_seq_mask: Tensor | None = None,  # [B] bool True when starting a new sequence
    ) -> tuple[Tensor, Tensor]:
        """Static-shape CUDA-Graph-compatible vLLM proposer step (0 bytes KV cache)."""
        if self.config.kvn_backbone_type != "kvn" or self.rollout is None:
            raise RuntimeError(
                "advance_and_draft_step requires kvn_backbone_type == 'kvn'."
            )

        bsz, s_max = token_ids_block.shape
        tok_emb = self.embed_tokens(token_ids_block)
        x_cur = prev_fc_block.to(tok_emb.dtype) + tok_emb

        step_indices = torch.arange(s_max, device=x_cur.device).unsqueeze(0)
        active_mask = step_indices < valid_lens.to(x_cur.device).unsqueeze(1)

        new_layer_states: list[Tensor] = []
        q_anchors_per_layer: list[Tensor] = []
        p_anchors_per_layer: list[Tensor] = []
        dtheta_anchors_per_layer: list[Tensor] = []

        for l_idx, kvn_layer in enumerate(self.kvn_layers):
            q_state = state_cache[l_idx, 0]
            p_state = state_cache[l_idx, 1]
            if reset_seq_mask is not None:
                keep_seq = (~reset_seq_mask.to(x_cur.device)).to(q_state.dtype).unsqueeze(-1)
                q_state = q_state * keep_seq
                p_state = p_state * keep_seq

            dtheta_anchor = torch.zeros(
                (bsz, kvn_layer.n_modes), dtype=x_cur.dtype, device=x_cur.device
            )
            h_steps: list[Tensor] = []

            for s in range(s_max):
                is_valid = active_mask[:, s].unsqueeze(-1)
                is_anchor = (valid_lens.to(x_cur.device) == (s + 1)).unsqueeze(-1)
                h_s, (q_cand, p_cand), dtheta_s = kvn_layer.recurrent_step(
                    x_cur[:, s : s + 1, :],
                    state=(q_state.unsqueeze(1), p_state.unsqueeze(1)),
                )
                q_state = torch.where(
                    is_valid, q_cand.squeeze(1).to(q_state.dtype), q_state
                )
                p_state = torch.where(
                    is_valid, p_cand.squeeze(1).to(p_state.dtype), p_state
                )
                dtheta_anchor = torch.where(
                    is_anchor, dtheta_s.squeeze(1).to(dtheta_anchor.dtype), dtheta_anchor
                )
                h_steps.append(h_s)

            new_layer_states.append(torch.stack([q_state, p_state], dim=0))
            q_anchors_per_layer.append(q_state.to(x_cur.dtype))
            p_anchors_per_layer.append(p_state.to(x_cur.dtype))
            dtheta_anchors_per_layer.append(dtheta_anchor)
            x_cur = torch.cat(h_steps, dim=1)

        state_cache.copy_(torch.stack(new_layer_states, dim=0))

        last_step = (valid_lens.to(x_cur.device) - 1).clamp(min=0)
        gather_h_idx = last_step.view(bsz, 1, 1).expand(bsz, 1, x_cur.size(-1))
        anchor_h = x_cur.gather(dim=1, index=gather_h_idx).squeeze(1)
        anchor_token_ids = token_ids_block.gather(
            dim=1, index=last_step.view(bsz, 1)
        ).squeeze(1)

        anchor_q, anchor_p, anchor_dtheta = self.rollout.aggregate_multilayer_states(
            q_anchors_per_layer, p_anchors_per_layer, dtheta_anchors_per_layer
        )
        raw_blocks, born_features = self.rollout(
            anchor_h=anchor_h,
            anchor_q=anchor_q,
            anchor_p=anchor_p,
            anchor_dtheta=anchor_dtheta,
        )
        if self.intrablock_mixer is not None:
            raw_blocks = self.intrablock_mixer(raw_blocks)

        hidden_blocks = self.norm(raw_blocks)
        base_logits = self.lm_head(hidden_blocks)
        return self._decode_draft_block_greedy(
            hidden_blocks=hidden_blocks,
            base_logits=base_logits,
            anchor_token_ids=anchor_token_ids,
            born_features=born_features,
        )

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        loss_mask: Tensor,
        verifier_last_hidden_states: Tensor,
        document_ids: Tensor,
        position_ids: Tensor | None = None,
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[Tensor, Tensor], Tensor] = _DEFAULT_TV_FN,
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

        confidence_logits = None
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

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

        loss, metrics = compute_metrics(
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
        )
        if self.training:
            lasso_pen = self.total_lasso_penalty(lasso_weight=kvn_lasso_weight)
            loss = loss + lasso_pen.to(loss.dtype)
        else:
            lasso_pen = torch.zeros((), device=loss.device, dtype=torch.float32)
        metrics["lasso_loss_sum"] = lasso_pen.detach().to(torch.float32)
        metrics["lasso_loss_total"] = torch.ones((), device=loss.device)
        return None, loss, metrics

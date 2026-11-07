from __future__ import annotations

from collections.abc import Callable
import importlib.util
from typing import ClassVar

import torch
from torch import nn
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
    MultiHorizonKoopmanRollout,
)
from speculators.models.utils import conditional_torch_compile

_HAS_TRITON = importlib.util.find_spec("triton") is not None
_DEFAULT_KL_FN = kl_div_loss if _HAS_TRITON else eager_losses.kl_div_loss
_DEFAULT_TV_FN = tv_loss if _HAS_TRITON else eager_losses.tv_loss
_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (_DEFAULT_KL_FN, 1.0)}

__all__ = [
    "KvNDSparkDraftModel",
]


@SpeculatorModel.register("kvn_dspark")
class KvNDSparkDraftModel(DSparkDraftModel):
    """Zero-KV-Cache Koopman-von Neumann RFF DSpark Speculator Model.

    Supports three controlled scientific variants via ``config.kvn_backbone_type``:
    - ``"kvn"``: 100% Zero-KV-Cache causal KvN-RFF oscillator layers + 1-pass
      Multi-Horizon Koopman Phase Rollout + MarkovHead + BornConfidenceHead.
    - ``"hybrid_1swa_2kvn"``: 1 SWA-128 attention layer + (L-1) KvN-RFF layers.
    - ``"swa"``: Standard DSpark SWA attention stack (baseline).
    """

    config_class: ClassVar[type[KvNDSparkSpeculatorConfig]] = KvNDSparkSpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlashDecoderLayer", "CausalKvNRFFLayer"]

    def __init__(self, config: KvNDSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        tl_config = config.transformer_layer_config
        hidden_size = tl_config.hidden_size
        num_draft_layers = tl_config.num_hidden_layers

        self.kvn_layers = nn.ModuleList()
        self.rollout: MultiHorizonKoopmanRollout | None = None

        if config.kvn_backbone_type == "kvn":
            # Eliminate 100% of SWA attention layers and KV cache
            self.layers = nn.ModuleList()
            self.sliding_window_indices = []
            self.uses_sliding_window_attn = False
            self.uses_full_attn = False
            for _ in range(num_draft_layers):
                self.kvn_layers.append(
                    CausalKvNRFFLayer(
                        d_model=hidden_size,
                        n_modes=config.kvn_n_modes,
                        reupload_layers_l=config.kvn_reupload_layers,
                        sigma_g=config.kvn_sigma_g,
                    )
                )
            self.rollout = MultiHorizonKoopmanRollout(
                d_model=hidden_size,
                n_modes=config.kvn_n_modes,
                block_size=self.block_size,
            )
        elif config.kvn_backbone_type == "hybrid_1swa_2kvn":
            # Keep only 1 SWA layer (layer 0) and replace remaining layers with KvN-RFF
            self.layers = nn.ModuleList(list(self.layers[:1]))
            self.sliding_window_indices = [
                i for i in self.sliding_window_indices if i < 1
            ]
            self.uses_sliding_window_attn = bool(self.sliding_window_indices)
            self.uses_full_attn = bool(len(self.layers) - len(self.sliding_window_indices))
            num_kvn = max(1, num_draft_layers - 1)
            for _ in range(num_kvn):
                self.kvn_layers.append(
                    CausalKvNRFFLayer(
                        d_model=hidden_size,
                        n_modes=config.kvn_n_modes,
                        reupload_layers_l=config.kvn_reupload_layers,
                        sigma_g=config.kvn_sigma_g,
                    )
                )
            self.rollout = MultiHorizonKoopmanRollout(
                d_model=hidden_size,
                n_modes=config.kvn_n_modes,
                block_size=self.block_size,
            )

        # Replace standard ConfidenceHead with BornConfidenceHead when enabled
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
            )
            self.confidence_head = None

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> KvNDSparkDraftModel:
        """Create a KvNDSparkDraftModel from training CLI/YAML arguments."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        kvn_use_born_arg = kwargs.get("kvn_use_born_confidence")

        config = KvNDSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("kvn_dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
            kvn_backbone_type=kwargs.get("kvn_backbone_type", "kvn"),
            kvn_n_modes=kwargs.get("kvn_n_modes", 256),
            kvn_reupload_layers=kwargs.get("kvn_reupload_layers", 2),
            kvn_sigma_g=kwargs.get("kvn_sigma_g", 1.0),
            kvn_lasso_weight=kwargs.get("kvn_lasso_weight", 1e-4),
            kvn_use_born_confidence=(
                True if kvn_use_born_arg is None else kvn_use_born_arg
            ),
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
        return dict(shared), dict(shared)

    def total_lasso_penalty(self) -> torch.Tensor:
        """Compute Barthe Theorem 5.6 Group-LASSO + L1 penalty across all KvN-RFF readouts."""
        if not self.kvn_layers or self.config.kvn_lasso_weight <= 0.0:
            return torch.tensor(0.0, device=self.fc.weight.device)
        pen = sum(layer.lasso_penalty() for layer in self.kvn_layers)
        if self.rollout is not None:
            pen = pen + self.rollout.lasso_penalty()
        if self.born_confidence_head is not None:
            pen = pen + self.born_confidence_head.lasso_penalty()
        return self.config.kvn_lasso_weight * pen

    @torch.no_grad()
    def prune_kvn_modes(self, target_modes: int) -> None:
        """Structurally prune all CausalKvNRFFLayer blocks (M_train -> M_infer) and align rollout/confidence heads."""
        if not self.kvn_layers:
            return
        last_top_idx: torch.Tensor | None = None
        num_kvn = len(self.kvn_layers)
        for idx, layer in enumerate(self.kvn_layers):
            extra_imp: torch.Tensor | None = None
            if idx == num_kvn - 1:
                if self.rollout is not None:
                    extra_imp = self.rollout.mode_importance()
                if self.born_confidence_head is not None:
                    born_imp = self.born_confidence_head.mode_importance()
                    if born_imp is not None:
                        extra_imp = (
                            born_imp
                            if extra_imp is None
                            else extra_imp + born_imp.to(extra_imp.device)
                        )
            last_top_idx = layer.prune_to_top_modes(
                target_modes, extra_importance=extra_imp
            )
        if last_top_idx is not None and self.rollout is not None:
            self.rollout.prune_to_indices(last_top_idx)
            if self.born_confidence_head is not None:
                self.born_confidence_head.prune_born_modes(last_top_idx)
        self.config.kvn_n_modes = target_modes

    @torch.compiler.disable
    def _select_kvn_anchors(
        self, loss_mask: torch.Tensor, max_anchors: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select valid anchors outside torch.compile to avoid data-dependent nonzero shapes."""
        return select_anchors(loss_mask, max_anchors, self.block_size)

    def _compute_targets_and_mask(
        self,
        verifier_last_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        document_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            == document_ids[:, anchor_positions].repeat_interleave(self.block_size, dim=1)
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
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        max_anchors: int = 512,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
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
            # 1 SWA layer first, then KvN-RFF layers + Koopman Rollout
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
            # 100% Zero-KV Pure KvN-RFF path (no SWA attention mask or KV cache)
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

        # Causal KvN-RFF evolution over fused verifier hidden states + token embeddings
        tok_emb = self.embed_tokens(input_ids).to(fc_output.dtype)
        x = fc_output + tok_emb

        q_last: torch.Tensor | None = None
        p_last: torch.Tensor | None = None
        dtheta_last: torch.Tensor | None = None
        for kvn_layer in self.kvn_layers:
            x, q_last, p_last, dtheta_last = kvn_layer(x)

        assert (
            self.rollout is not None
            and q_last is not None
            and p_last is not None
            and dtheta_last is not None
        )
        anchor_h = x[0, anchor_positions]
        hidden_blocks, born_features = self.rollout(
            anchor_h=anchor_h,
            anchor_q=q_last[0, anchor_positions],
            anchor_p=p_last[0, anchor_positions],
            anchor_dtheta=dtheta_last[0, anchor_positions],
        )
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

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = _DEFAULT_TV_FN,
        gamma: float = 4.0,
        max_anchors: int = 512,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
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
        lasso_pen = self.total_lasso_penalty()
        loss = loss + lasso_pen.to(loss.dtype)
        return None, loss, metrics

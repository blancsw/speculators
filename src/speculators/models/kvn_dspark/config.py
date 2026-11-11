from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig

__all__ = [
    "KvNDSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("kvn_dspark")
class KvNDSparkSpeculatorConfig(DSparkSpeculatorConfig):
    """Zero-Drafter-KV-Cache Dequantized Damped Bosonic SSM Speculator Configuration.

    Replaces DSpark's SWA-128 attention decoder layers with causal damped Bosonic
    oscillator layers driven by Modal Data-Reuploading RFFs (Barthe 2026, Thm. 2.1 &
    Cor. 3.2), a multi-scale 1-pass Multi-Horizon Koopman Rollout, an ephemeral
    Zero-KV IntraBlockCausalMixer, dyadic Group-LASSO mode pruning, and vacuum-mode
    interferometric Born confidence calibration (Barthe Thm. 4.2 Eq. 4.34 & Ch. 7).
    """

    speculators_model_type: Literal["kvn_dspark"] = "kvn_dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["KvNDSparkSpeculator"],
        description="Model architectures that can load these weights",
    )

    kvn_backbone_type: Literal["kvn", "hybrid_1swa_2kvn", "swa"] = Field(
        default="kvn",
        description=(
            "Backbone variant for controlled scientific ablation: "
            "'kvn' (Zero-KV pure damped Bosonic KvN-RFF), "
            "'hybrid_1swa_2kvn' (1 SWA layer + KvN-RFF layers), "
            "or 'swa' (standard DSpark SWA baseline)."
        ),
    )
    kvn_n_modes: int = Field(
        default=256,
        gt=0,
        description="Number of Bosonic RFF harmonic oscillator modes M per CausalKvNRFFLayer.",
    )
    kvn_reupload_layers: int = Field(
        default=2,
        gt=0,
        description="Number of Barthe Data Re-uploading stages L per CausalKvNRFFLayer.",
    )
    kvn_reupload_dim: int = Field(
        default=512,
        ge=0,
        description=(
            "Width d_re of the ModalDataReuploadingRFF spectral bottleneck "
            "(0 automatically resolves eff_d_re = d_model for full-width ablation)."
        ),
    )
    kvn_learnable_rff: bool = Field(
        default=True,
        description=(
            "Train RFF spectral frequencies omega_rff under AdamW to satisfy Barthe "
            "Theorem 2.1 spectral alignment."
        ),
    )
    kvn_sigma_g: float = Field(
        default=1.0,
        gt=0.0,
        description="Spectral bandwidth for Bosonic RFF initialization (scaled by 1/sqrt(L*d_re)).",
    )
    kvn_damping_min: float = Field(
        default=1e-3,
        gt=0.0,
        description="Minimum modal damping rate eta_min > 0 for open-system stability.",
    )
    kvn_damping_max: float = Field(
        default=2.0,
        gt=0.0,
        description="Maximum modal damping rate eta_max for damped Bosonic evolution.",
    )
    kvn_lasso_weight: float = Field(
        default=1e-4,
        ge=0.0,
        description="Dyadic Group-LASSO (L_{2,1}) + L1 penalty weight on RFF readout columns.",
    )
    kvn_use_born_confidence: bool = Field(
        default=True,
        description=(
            "Augment the DSpark confidence head with interferometric Born-rule energy "
            "observables."
        ),
    )
    kvn_enable_interferometer: bool = Field(
        default=True,
        description=(
            "Apply O(M) DyadicSymplecticInterferometer beamsplitters before Born "
            "detection to restore Koopman phase sensitivity."
        ),
    )
    kvn_vacuum_born_norm: bool = Field(
        default=True,
        description=(
            "Normalize Born energies on the (M+1)-simplex with a learnable vacuum "
            "mode |M+1> (Barthe Ch. 4 Thm. 4.2 Eq. 4.34)."
        ),
    )
    kvn_anchor_rank: int = Field(
        default=512,
        ge=0,
        description=(
            "Low-rank bottleneck r_a for MultiHorizonKoopmanRollout.anchor_proj "
            "(0 uses full dense d_model -> K * d_model projection)."
        ),
    )
    kvn_multilayer_rollout: bool = Field(
        default=True,
        description="Aggregate modal states (q^(l), p^(l)) across all draft layers in rollout.",
    )
    kvn_intrablock_mixer: bool = Field(
        default=True,
        description=(
            "Attach the Zero-KV IntraBlockCausalMixer (K x K) over speculative slots "
            "to capture multi-step intra-block dependencies."
        ),
    )
    kvn_strict_causal_shift: bool = Field(
        default=True,
        description=(
            "Shift verifier fc_output by 1 step (masked across document_ids boundaries) "
            "so anchor position a strictly observes fc_output[0..a-1] + input_ids[0..a]."
        ),
    )
    kvn_ffn_mult: float = Field(
        default=1.5,
        gt=0.0,
        description="SwiGLU/MLP hidden multiplier relative to hidden_size in CausalKvNRFFLayer.",
    )
    kvn_state_dtype: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Precision of the persistent recurrent (q, p) state buffer at inference.",
    )

    def _kv_hidden_dim(self) -> int:
        """Compute exact GQA KV hidden dimension: num_key_value_heads * head_dim."""
        tl_cfg = self.transformer_layer_config
        hidden_size = int(getattr(tl_cfg, "hidden_size", 1024))
        n_heads = int(getattr(tl_cfg, "num_attention_heads", 0) or 0)
        n_kv_heads = int(getattr(tl_cfg, "num_key_value_heads", 0) or n_heads)
        head_dim = getattr(tl_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = (hidden_size // n_heads) if n_heads > 0 else hidden_size
            if n_kv_heads == 0:
                return hidden_size
        return int(n_kv_heads * int(head_dim))

    def estimated_state_bytes_per_seq(
        self,
        dtype_bytes: int | None = None,
        window_size: int | None = None,
    ) -> int:
        """Compute exact per-sequence inference recurrent/KV state footprint in bytes."""
        tl_cfg = self.transformer_layer_config
        num_layers = int(getattr(tl_cfg, "num_hidden_layers", 3))
        sliding_window = int(
            window_size or getattr(tl_cfg, "sliding_window", 128) or 128
        )
        kv_dim = self._kv_hidden_dim()
        kv_elem_bytes = dtype_bytes if dtype_bytes is not None else 2
        state_elem_bytes = (
            dtype_bytes
            if dtype_bytes is not None
            else (4 if self.kvn_state_dtype == "float32" else 2)
        )

        if self.kvn_backbone_type == "swa":
            return num_layers * 2 * sliding_window * kv_dim * kv_elem_bytes
        if self.kvn_backbone_type == "hybrid_1swa_2kvn":
            swa_bytes = 1 * 2 * sliding_window * kv_dim * kv_elem_bytes
            kvn_bytes = max(1, num_layers - 1) * 2 * self.kvn_n_modes * state_elem_bytes
            return swa_bytes + kvn_bytes
        return num_layers * 2 * self.kvn_n_modes * state_elem_bytes

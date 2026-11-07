from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig

__all__ = [
    "KvNDSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("kvn_dspark")
class KvNDSparkSpeculatorConfig(DSparkSpeculatorConfig):
    """Zero-KV-Cache Koopman-von Neumann RFF DSpark Speculator Configuration.

    Replaces DSpark's SWA-128 attention decoder layers with causal symplectic
    Koopman-von Neumann Random Fourier Feature (KvN-RFF) harmonic oscillators and
    a 1-pass Multi-Horizon Koopman Phase Rollout, plus Barthe Theorem 5.6 LASSO
    mode pruning and Barthe Theorem 4.2 amplified Born-rule confidence features.
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
            "'kvn' (Zero-KV pure Koopman-von Neumann RFF), "
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
    kvn_sigma_g: float = Field(
        default=1.0,
        gt=0.0,
        description="Spectral bandwidth for Bosonic RFF initialization.",
    )
    kvn_lasso_weight: float = Field(
        default=1e-4,
        ge=0.0,
        description="Barthe Theorem 5.6 LASSO L1 penalty weight on RFF readout columns.",
    )
    kvn_use_born_confidence: bool = Field(
        default=True,
        description=(
            "Augment the DSpark confidence head with Barthe Theorem 4.2 amplified "
            "Born-rule energy observables 2M*(q^2 + p^2) - 1."
        ),
    )

    def estimated_state_bytes_per_seq(self, dtype_bytes: int = 2) -> int:
        """Compute per-sequence inference recurrent/KV state footprint in bytes."""
        tl_cfg = self.transformer_layer_config
        hidden_size = getattr(tl_cfg, "hidden_size", 1024)
        num_layers = getattr(tl_cfg, "num_hidden_layers", 3)
        sliding_window = getattr(tl_cfg, "sliding_window", 128) or 128
        if self.kvn_backbone_type == "swa":
            return num_layers * 2 * sliding_window * hidden_size * dtype_bytes
        if self.kvn_backbone_type == "hybrid_1swa_2kvn":
            swa_bytes = 1 * 2 * sliding_window * hidden_size * dtype_bytes
            kvn_bytes = max(1, num_layers - 1) * 2 * self.kvn_n_modes * dtype_bytes
            return swa_bytes + kvn_bytes
        return num_layers * 2 * self.kvn_n_modes * dtype_bytes

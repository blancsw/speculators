from typing import Any, Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig

__all__ = [
    "KvNDSparkSpeculatorConfig",
]

_KVN_V3_ALIAS_MAP: dict[str, str] = {
    "kvn_dyadic_rff": "dyadic_rff",
    "kvn_use_engram": "use_engram",
    "kvn_engram_buckets": "engram_buckets",
    "kvn_engram_dim": "engram_dim",
    "kvn_symplectic_butterfly": "use_symplectic_butterfly",
    "kvn_use_symplectic_butterfly": "use_symplectic_butterfly",
    "kvn_squeezing_max_ratio": "squeezing_max_ratio",
    "kvn_lindblad_purity": "use_lindblad_purity",
    "kvn_use_lindblad_purity": "use_lindblad_purity",
    "kvn_use_kraus_tree": "use_kraus_tree",
    "kvn_kraus_top_c": "kraus_top_c",
    "kvn_use_triton_scan": "use_triton_scan",
    "kvn_self_cond_prob": "self_cond_rollout_prob",
    "kvn_self_cond_rollout_prob": "self_cond_rollout_prob",
}


@SpeculatorModelConfig.register("kvn_dspark")
class KvNDSparkSpeculatorConfig(DSparkSpeculatorConfig):
    """Zero-KV-Cache Symplectic-Squeezed Bosonic SSM Speculator Config (v3).

    Replaces DSpark's SWA-128 attention decoder layers with causal driven-damped
    Bosonic oscillator layers driven by Multiplicative Dyadic Modal Data-Reuploading
    RFFs (Barthe 2026, Ch. 2 Thm. 2.1 & Ch. 3 Lem. 3.2 Eq. 3.26), sp(4, R) butterfly
    symplectic-squeezed scans (Barthe Ch. 7 Eq. 7.2, 7.25, 7.56), Lindblad open-system
    thermal covariance & purity, Multi-Horizon Koopman Rollout, Zero-KV
    IntraBlockCausalMixer, hashed SpeculativeEngram memory, proximal dyadic Group-LASSO
    mode pruning (Barthe Ch. 5 Thm. 5.6), vacuum-mode Born calibration (Barthe Ch. 4
    Thm. 4.2 Eq. 4.34), and diagonal Kraus POVM speculative trees.
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
            "'kvn' (Zero-KV pure driven-damped Bosonic SSM), "
            "'hybrid_1swa_2kvn' (sequential (L-1) KvN layers -> 1 SWA layer), "
            "or 'swa' (standard DSpark SWA baseline)."
        ),
    )
    kvn_n_modes: int = Field(
        default=256,
        gt=0,
        description="Bosonic RFF harmonic oscillator modes M per CausalKvNRFFLayer.",
    )
    kvn_reupload_layers: int = Field(
        default=2,
        gt=0,
        description="Barthe Data Re-uploading stages L per CausalKvNRFFLayer.",
    )
    kvn_reupload_dim: int = Field(
        default=512,
        ge=0,
        description=(
            "Width d_re of ModalDataReuploadingRFF spectral bottleneck "
            "(0 resolves full d_model, or DataReuploadingLinear when dyadic_rff=False)."
        ),
    )
    kvn_learnable_rff: bool = Field(
        default=True,
        description=(
            "Train RFF spectral frequencies omega_rff under AdamW to satisfy Barthe "
            "Ch. 2 Theorem 2.1 spectral alignment."
        ),
    )
    kvn_sigma_g: float = Field(
        default=1.0,
        gt=0.0,
        description="Spectral bandwidth for Bosonic RFF initialization.",
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
        description="Dyadic Group-LASSO (L_{2,1}) + L1 penalty weight on RFF readout.",
    )
    kvn_use_born_confidence: bool = Field(
        default=True,
        description=(
            "Augment the DSpark confidence head with interferometric Born-rule energy, "
            "vacuum probability p_vac, and Lindblad mode purity."
        ),
    )
    kvn_enable_interferometer: bool = Field(
        default=True,
        description=(
            "Apply O(M) DyadicSymplecticInterferometer beamsplitters (Barthe Eq. 7.25) "
            "before Born detection to restore Koopman phase sensitivity."
        ),
    )
    kvn_vacuum_born_norm: bool = Field(
        default=True,
        description=(
            "Normalize Born energies on the dyadic simplex with a learnable vacuum "
            "mode (Barthe Ch. 4 Thm. 4.2 Eq. 4.34)."
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
        description="Aggregate modal states across all draft layers in rollout.",
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
            "Shift verifier fc_output by 1 step (masked across doc boundaries) "
            "so anchor a strictly observes fc_output[0..a-1] + input_ids[0..a]."
        ),
    )
    kvn_ffn_mult: float = Field(
        default=1.5,
        gt=0.0,
        description="SwiGLU/MLP hidden multiplier vs hidden_size in CausalKvNRFFLayer.",
    )
    kvn_state_dtype: Literal["bfloat16", "float32"] = Field(
        default="float32",
        description=(
            "Precision of persistent recurrent (q, p[, nu]) state buffer "
            "(FP32 default for exact long-context parity)."
        ),
    )

    # KvN-DSpark v3 Quantum-Dynamical Bricks (Axes 1, 2, 3, 4, 5, 6)
    dyadic_rff: bool = Field(
        default=True,
        description=(
            "Compose L re-uploading stages multiplicatively in complex phase with "
            "dyadic 2^l frequency scaling (Barthe Ch. 3 Lem. 3.2 Eq. 3.26)."
        ),
    )
    use_engram: bool = Field(
        default=True,
        description="Attach O(1) hashed 2-gram & 3-gram SpeculativeEngram residual.",
    )
    engram_buckets: int = Field(
        default=65536,
        gt=0,
        description="Number of hash buckets in SpeculativeEngram embedding table.",
    )
    engram_dim: int = Field(
        default=128,
        gt=0,
        description="Embedding width of SpeculativeEngram table.",
    )
    use_symplectic_butterfly: bool = Field(
        default=True,
        description=(
            "Use sp(4, R) butterfly symplectic-squeezed 4x4 dyad scan with selective "
            "beamsplitters and bounded active squeezing."
        ),
    )
    squeezing_max_ratio: float = Field(
        default=0.8,
        ge=0.0,
        lt=1.0,
        description="Max squeezing-to-damping ratio |xi|/gamma < 1 for ||A||_2 < 1.",
    )
    use_lindblad_purity: bool = Field(
        default=True,
        description="Track Lindblad excess thermal variance delta_nu >= 0 and purity.",
    )
    use_kraus_tree: bool = Field(
        default=True,
        description="Enable diagonal Kraus POVM head for speculative tree expansion.",
    )
    kraus_top_c: int = Field(
        default=256,
        gt=0,
        description="Candidate shortlist size C for active Kraus POVM normalization.",
    )
    use_triton_scan: bool = Field(
        default=True,
        description="Dispatch CUDA scans to Mega-KvN Triton kernels when available.",
    )
    self_cond_rollout_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of 2-pass stop-grad self-conditioned Markov rollout.",
    )

    def __init__(self, **kwargs: Any) -> None:
        normalized = self._normalize_kvn_v3_aliases(kwargs)
        super().__init__(**normalized)

    @model_validator(mode="before")
    @classmethod
    def _normalize_kvn_v3_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "speculators_config" in out and out["speculators_config"] is None:
            out.pop("speculators_config")
        for alias_key, canonical_key in _KVN_V3_ALIAS_MAP.items():
            if alias_key in out:
                alias_val = out.pop(alias_key)
                if canonical_key not in out and alias_val is not None:
                    out[canonical_key] = alias_val
        return out

    @model_validator(mode="after")
    def _validate_dyadic_mode_parity(self) -> "KvNDSparkSpeculatorConfig":
        if (
            self.kvn_backbone_type != "swa"
            and (
                self.use_symplectic_butterfly
                or self.kvn_enable_interferometer
                or self.kvn_vacuum_born_norm
            )
            and self.kvn_n_modes % 2 != 0
        ):
            raise ValueError(
                "kvn_n_modes must be even when dyadic/symplectic modules are active, "
                f"got {self.kvn_n_modes}."
            )
        return self

    @property
    def kvn_dyadic_rff(self) -> bool:
        return self.dyadic_rff

    @kvn_dyadic_rff.setter
    def kvn_dyadic_rff(self, value: bool) -> None:
        self.dyadic_rff = bool(value)

    @property
    def kvn_use_engram(self) -> bool:
        return self.use_engram

    @kvn_use_engram.setter
    def kvn_use_engram(self, value: bool) -> None:
        self.use_engram = bool(value)

    @property
    def kvn_engram_buckets(self) -> int:
        return self.engram_buckets

    @kvn_engram_buckets.setter
    def kvn_engram_buckets(self, value: int) -> None:
        self.engram_buckets = int(value)

    @property
    def kvn_engram_dim(self) -> int:
        return self.engram_dim

    @kvn_engram_dim.setter
    def kvn_engram_dim(self, value: int) -> None:
        self.engram_dim = int(value)

    @property
    def kvn_symplectic_butterfly(self) -> bool:
        return self.use_symplectic_butterfly

    @kvn_symplectic_butterfly.setter
    def kvn_symplectic_butterfly(self, value: bool) -> None:
        self.use_symplectic_butterfly = bool(value)

    @property
    def kvn_squeezing_max_ratio(self) -> float:
        return self.squeezing_max_ratio

    @kvn_squeezing_max_ratio.setter
    def kvn_squeezing_max_ratio(self, value: float) -> None:
        self.squeezing_max_ratio = float(value)

    @property
    def kvn_lindblad_purity(self) -> bool:
        return self.use_lindblad_purity

    @kvn_lindblad_purity.setter
    def kvn_lindblad_purity(self, value: bool) -> None:
        self.use_lindblad_purity = bool(value)

    @property
    def kvn_use_kraus_tree(self) -> bool:
        return self.use_kraus_tree

    @kvn_use_kraus_tree.setter
    def kvn_use_kraus_tree(self, value: bool) -> None:
        self.use_kraus_tree = bool(value)

    @property
    def kvn_kraus_top_c(self) -> int:
        return self.kraus_top_c

    @kvn_kraus_top_c.setter
    def kvn_kraus_top_c(self, value: int) -> None:
        self.kraus_top_c = int(value)

    @property
    def kvn_use_triton_scan(self) -> bool:
        return self.use_triton_scan

    @kvn_use_triton_scan.setter
    def kvn_use_triton_scan(self, value: bool) -> None:
        self.use_triton_scan = bool(value)

    @property
    def kvn_self_cond_prob(self) -> float:
        return self.self_cond_rollout_prob

    @kvn_self_cond_prob.setter
    def kvn_self_cond_prob(self, value: float) -> None:
        self.self_cond_rollout_prob = float(value)

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

    def num_recurrent_state_vars(self) -> int:
        """Return real scalars stored per Bosonic mode: 3 (q, p, nu) or 2 (q, p)."""
        return 3 if self.use_lindblad_purity else 2

    def estimated_state_bytes_breakdown(
        self,
        dtype_bytes: int | None = None,
        window_size: int | None = None,
    ) -> dict[str, int]:
        """Return per-sequence serving state breakdown (oscillator + fc_{t-1})."""
        tl_cfg = self.transformer_layer_config
        num_layers = int(getattr(tl_cfg, "num_hidden_layers", 3))
        hidden_size = int(getattr(tl_cfg, "hidden_size", 1024))
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
        n_vars = self.num_recurrent_state_vars()
        swa_equiv = num_layers * 2 * sliding_window * kv_dim * kv_elem_bytes

        if self.kvn_backbone_type == "swa":
            return {
                "recurrent_state_bytes": swa_equiv,
                "fc_boundary_cache_bytes": 0,
                "total_serving_bytes": swa_equiv,
                "swa_kv_equivalent_bytes": swa_equiv,
            }

        fc_cache_bytes = (
            hidden_size * kv_elem_bytes if self.kvn_strict_causal_shift else 0
        )
        if self.kvn_backbone_type == "hybrid_1swa_2kvn":
            swa_bytes = 1 * 2 * sliding_window * kv_dim * kv_elem_bytes
            kvn_bytes = (
                max(1, num_layers - 1) * n_vars * self.kvn_n_modes * state_elem_bytes
            )
            rec_bytes = swa_bytes + kvn_bytes
        else:
            rec_bytes = num_layers * n_vars * self.kvn_n_modes * state_elem_bytes

        return {
            "recurrent_state_bytes": rec_bytes,
            "fc_boundary_cache_bytes": fc_cache_bytes,
            "total_serving_bytes": rec_bytes + fc_cache_bytes,
            "swa_kv_equivalent_bytes": swa_equiv,
        }

    def estimated_state_bytes_per_seq(
        self,
        dtype_bytes: int | None = None,
        window_size: int | None = None,
        include_fc_boundary_cache: bool = False,
    ) -> int:
        """Compute exact per-sequence recurrent/KV state footprint in bytes."""
        bd = self.estimated_state_bytes_breakdown(
            dtype_bytes=dtype_bytes, window_size=window_size
        )
        if include_fc_boundary_cache:
            return bd["total_serving_bytes"]
        return bd["recurrent_state_bytes"]

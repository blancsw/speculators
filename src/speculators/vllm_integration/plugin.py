"""``vllm.general_plugins`` entrypoint for Zero-KV-Cache ``KvN-DSpark`` drafting.

Registered under ``[project.entry-points."vllm.general_plugins"]`` in ``pyproject.toml``
so ``vllm serve <target> --speculative-config '{"model": "<kvn_ckpt>"}'`` automatically:
1. Registers ``speculators_model_type = "kvn_dspark"`` in vLLM's ``SpeculatorsConfig``
   ``SUPPORTED_SPECULATORS_TYPES`` registry via ``update_kvn_dspark_pretrained_config``
   and patches ``SpeculatorsConfig.build_vllm_speculative_config``,
   ``EngineArgs.create_speculative_config``, and ``SpeculativeConfig`` initialization
   (routing ``kvn`` / ``hybrid_1swa_2kvn`` checkpoints to ``KvNDSparkVllmProposer`` via
   ``method="custom_class"`` and ``swa`` checkpoints to native ``method="dspark"`` with
   exact ``num_speculative_tokens``);
2. Enables ``use_aux_hidden_state_outputs = True`` on ``GPUModelRunner``, preserves
   ``KvNDSparkDraftModel.aux_hidden_state_layer_ids`` inside
   ``_get_eagle3_aux_layers_from_config()`` prior to CUDA Graph capture, differentiates
   ``SpeculativeConfig.compute_hash()`` for ``torch.compile`` caching, and routes
   ``propose_draft_token_ids`` via ``KvNDSparkVllmProposer.propose_vllm_step``;
3. Cleanly returns ``False`` with zero side effects when ``vllm`` is not installed.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PROPOSER_IMPORT_PATH",
    "configure_kvn_speculator_dict",
    "is_kvn_vllm_plugin_registered",
    "register_kvn_vllm_plugin",
    "update_kvn_dspark_pretrained_config",
]

PROPOSER_IMPORT_PATH = "speculators.vllm_integration.kvn_proposer.KvNDSparkVllmProposer"
_PLUGIN_REGISTERED = False
_PATCHED_CLASSES: set[type[Any]] = set()
_PRESERVED_KVN_EXTRA_KEYS = frozenset(
    {
        "use_lindblad_purity",
        "use_kraus_head",
        "kraus_rank",
        "kraus_logit_rank",
        "aux_hidden_state_layer_ids",
        "speculators_model_type",
    }
)
_DSPARK_FALLBACK_KEYS = (
    "draft_vocab_size",
    "target_hidden_size",
    "mask_token_id",
    "markov_rank",
    "markov_head_type",
    "block_size",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "use_aux_hidden_state",
)


def is_kvn_vllm_plugin_registered() -> bool:
    """Return whether ``register_kvn_vllm_plugin`` patched the active vLLM runtime."""
    return _PLUGIN_REGISTERED


def _compute_default_num_spec_tokens(config_dict: dict[str, Any]) -> int:
    """Compute default speculative horizon from ``block_size`` and anchor mode."""
    block_size = int(config_dict.get("block_size", 16))
    sample_from_anchor = bool(config_dict.get("sample_from_anchor", False))
    return block_size if sample_from_anchor else max(1, block_size - 1)


def _apply_dspark_fallback_fields(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Populate DSpark fields when ``vllm`` is absent (standalone unit tests)."""
    sample_from_anchor = bool(config_dict.get("sample_from_anchor", False))
    pre_trained_config["architectures"] = ["Qwen3DSparkModel"]
    pre_trained_config["sample_from_anchor"] = sample_from_anchor
    pre_trained_config["dspark_bonus_anchor"] = not sample_from_anchor
    aux_layer_ids = list(config_dict.get("aux_hidden_state_layer_ids") or [])
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["target_layer_ids"] = [int(i) - 1 for i in aux_layer_ids]
    for key in _DSPARK_FALLBACK_KEYS:
        if config_dict.get(key) is not None:
            pre_trained_config[key] = config_dict[key]


def update_kvn_dspark_pretrained_config(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Populate ``pre_trained_config`` from a ``KvNDSpark`` checkpoint ``config.json``.

    Registered under ``SUPPORTED_SPECULATORS_TYPES["kvn_dspark"]`` in
    ``vllm.transformers_utils.configs.speculators.algos`` where
    ``SpeculatorsConfig.extract_transformers_pre_trained_config`` invokes
    ``algo_updater(config_dict=config_dict, pre_trained_config=pre_trained_config)``.
    """
    update_dspark = None
    try:
        algos_mod = importlib.import_module(
            "vllm.transformers_utils.configs.speculators.algos"
        )
        update_dspark = getattr(algos_mod, "update_dspark", None)
    except ImportError:
        update_dspark = None

    if callable(update_dspark):
        update_dspark(config_dict=config_dict, pre_trained_config=pre_trained_config)
    else:
        _apply_dspark_fallback_fields(config_dict, pre_trained_config)

    for key, val in config_dict.items():
        if val is not None and (
            key.startswith("kvn_") or key in _PRESERVED_KVN_EXTRA_KEYS
        ):
            pre_trained_config[key] = val


def configure_kvn_speculator_dict(
    config_dict: dict[str, Any],
    vllm_speculative_config: dict[str, Any],
) -> str | None:
    """Populate ``vllm_speculative_config`` from a ``KvNDSpark`` config dict.

    Preserves an explicitly provided ``num_speculative_tokens`` if already present and
    not ``None``. Returns the resolved ``kvn_checkpoint_path`` string when routing to
    ``method="custom_class"`` (or ``None`` for ``swa`` -> ``method="dspark"``).
    """
    backbone = str(config_dict.get("kvn_backbone_type", "kvn"))
    explicit_num_spec = vllm_speculative_config.get("num_speculative_tokens")
    num_spec = (
        int(explicit_num_spec)
        if explicit_num_spec is not None
        else _compute_default_num_spec_tokens(config_dict)
    )

    if backbone == "swa":
        vllm_speculative_config["method"] = "dspark"
        vllm_speculative_config["num_speculative_tokens"] = num_spec
        return None

    ckpt_raw = vllm_speculative_config.get("model")
    resolved_ckpt: str | None = None
    if isinstance(ckpt_raw, os.PathLike):
        ckpt_raw = str(ckpt_raw)
    if isinstance(ckpt_raw, str) and ckpt_raw and ckpt_raw != PROPOSER_IMPORT_PATH:
        resolved_ckpt = ckpt_raw
        os.environ["KVN_DSPARK_CHECKPOINT_PATH"] = resolved_ckpt

    vllm_speculative_config["method"] = "custom_class"
    vllm_speculative_config["model"] = PROPOSER_IMPORT_PATH
    vllm_speculative_config["num_speculative_tokens"] = num_spec
    return resolved_ckpt


def _resolve_config_json_path(model_str: str) -> tuple[str, Path] | None:
    """Resolve ``(checkpoint_dir_str, config_json_path)`` if ``model_str`` is local."""
    candidate = Path(model_str)
    if candidate.is_dir():
        cfg_file = candidate / "config.json"
        return (model_str, cfg_file) if cfg_file.is_file() else None
    if candidate.is_file() and candidate.name == "config.json":
        return str(candidate.parent), candidate
    return None


def _load_kvn_ckpt_config_dict(model_ref: Any) -> tuple[str, dict[str, Any]] | None:
    """If ``model_ref`` points to a local ``KvNDSpark`` checkpoint, load its config."""
    if not isinstance(model_ref, (str, os.PathLike)):
        return None
    model_str = str(model_ref)
    if not model_str or model_str == PROPOSER_IMPORT_PATH:
        return None

    try:
        resolved_pair = _resolve_config_json_path(model_str)
        if resolved_pair is None:
            return None
        ckpt_dir_str, cfg_file = resolved_pair
        cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None

    if not isinstance(cfg_dict, dict):
        return None
    architectures = cfg_dict.get("architectures") or []
    is_kvn = (
        cfg_dict.get("speculators_model_type") == "kvn_dspark"
        or "KvNDSparkDraftModel" in architectures
        or "kvn_backbone_type" in cfg_dict
    )
    return (ckpt_dir_str, cfg_dict) if is_kvn else None


def _maybe_rewrite_kvn_speculative_dict(spec_dict: dict[str, Any]) -> str | None:
    """Inspect ``spec_dict["model"]`` and rewrite in-place if it is a KvN checkpoint."""
    loaded = _load_kvn_ckpt_config_dict(spec_dict.get("model"))
    if loaded is None:
        return None
    ckpt_dir_str, cfg_dict = loaded
    spec_dict["model"] = ckpt_dir_str
    return configure_kvn_speculator_dict(cfg_dict, spec_dict)


def _patch_speculators_algos_and_base() -> None:
    """Register ``kvn_dspark`` in ``speculators.algos`` and ``SpeculatorsConfig``."""
    algos_mod = importlib.import_module(
        "vllm.transformers_utils.configs.speculators.algos"
    )
    supported = getattr(algos_mod, "SUPPORTED_SPECULATORS_TYPES", None)
    if isinstance(supported, dict):
        supported["kvn_dspark"] = update_kvn_dspark_pretrained_config

    base_mod = importlib.import_module(
        "vllm.transformers_utils.configs.speculators.base"
    )
    spec_cfg_cls = getattr(base_mod, "SpeculatorsConfig", None)
    if spec_cfg_cls is None or spec_cfg_cls in _PATCHED_CLASSES:
        return

    orig_build_fn = spec_cfg_cls.__dict__.get("build_vllm_speculative_config")
    orig_callable = getattr(spec_cfg_cls, "build_vllm_speculative_config", None)
    if callable(orig_callable):

        @classmethod  # type: ignore[misc]
        def _wrapped_build_vllm_spec_cfg(
            cls: type[Any],
            config_dict: dict[str, Any],
        ) -> dict[str, Any]:
            if config_dict.get("speculators_model_type") == "kvn_dspark":
                spec_out: dict[str, Any] = {}
                proposal_methods = (config_dict.get("speculators_config") or {}).get(
                    "proposal_methods"
                ) or []
                if proposal_methods and isinstance(proposal_methods[0], dict):
                    pm_tokens = proposal_methods[0].get("speculative_tokens")
                    if pm_tokens is not None:
                        spec_out["num_speculative_tokens"] = int(pm_tokens)
                configure_kvn_speculator_dict(config_dict, spec_out)
                return spec_out
            if isinstance(orig_build_fn, classmethod):
                return orig_build_fn.__func__(cls, config_dict)
            return orig_callable(config_dict)

        spec_cfg_cls.build_vllm_speculative_config = _wrapped_build_vllm_spec_cfg
    _PATCHED_CLASSES.add(spec_cfg_cls)


def _merge_positional_spec_args(
    orig_sig: inspect.Signature | None,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Convert positional ``SpeculativeConfig`` args to kwargs when ``model`` given."""
    if "model" in kwargs or not args or orig_sig is None:
        return args, kwargs
    try:
        bound = orig_sig.bind_partial(instance, *args, **kwargs)
    except TypeError:
        return args, kwargs
    if "model" not in bound.arguments:
        return args, kwargs
    merged = dict(bound.arguments)
    merged.pop(next(iter(orig_sig.parameters)), None)
    return (), merged


def _compute_kvn_spec_hash(spec_instance: Any, base_hash: str) -> str:
    """Derive a ``torch.compile``-unique SHA-256 hash for KvN ``custom_class`` runs."""
    ckpt_attr = getattr(spec_instance, "kvn_checkpoint_path", None)
    if getattr(spec_instance, "model", None) != PROPOSER_IMPORT_PATH and (
        ckpt_attr is None
    ):
        return base_hash
    ckpt_id = ckpt_attr or os.environ.get("KVN_DSPARK_CHECKPOINT_PATH") or ""
    loaded_info = _load_kvn_ckpt_config_dict(ckpt_id)
    aux_ids: tuple[int, ...] = ()
    if loaded_info is not None:
        raw_aux = loaded_info[1].get("aux_hidden_state_layer_ids") or []
        aux_ids = tuple(int(i) for i in raw_aux)
    payload = repr((base_hash, "kvn_dspark", str(ckpt_id), aux_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _patch_speculative_config_cls(spec_cls: type[Any]) -> None:
    """Wrap ``SpeculativeConfig.__init__`` and ``SpeculativeConfig.compute_hash``."""
    if spec_cls in _PATCHED_CLASSES:
        return
    orig_spec_init = spec_cls.__init__
    try:
        orig_sig: inspect.Signature | None = inspect.signature(orig_spec_init)
    except (TypeError, ValueError):
        orig_sig = None

    @functools.wraps(orig_spec_init)
    def _wrapped_spec_init(self: Any, *args: Any, **kwargs: Any) -> None:
        call_args, call_kwargs = _merge_positional_spec_args(
            orig_sig, self, args, dict(kwargs)
        )
        normalized = {k.replace("-", "_"): v for k, v in call_kwargs.items()}
        resolved_ckpt = _maybe_rewrite_kvn_speculative_dict(normalized)
        orig_spec_init(self, *call_args, **normalized)
        if resolved_ckpt is not None:
            object.__setattr__(self, "kvn_checkpoint_path", resolved_ckpt)

    spec_cls.__init__ = _wrapped_spec_init  # type: ignore[method-assign]

    orig_compute_hash = getattr(spec_cls, "compute_hash", None)
    if callable(orig_compute_hash):

        @functools.wraps(orig_compute_hash)
        def _wrapped_compute_hash(self: Any) -> str:
            return _compute_kvn_spec_hash(self, orig_compute_hash(self))

        spec_cls.compute_hash = _wrapped_compute_hash

    _PATCHED_CLASSES.add(spec_cls)


def _patch_engine_args_cls(engine_args_cls: type[Any]) -> None:
    """Wrap ``EngineArgs.create_speculative_config`` for ``--speculative-config``."""
    if engine_args_cls in _PATCHED_CLASSES:
        return
    orig_create_spec = getattr(engine_args_cls, "create_speculative_config", None)
    if callable(orig_create_spec):

        @functools.wraps(orig_create_spec)
        def _wrapped_create_speculative_config(
            self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            resolved_ckpt: str | None = None
            raw_spec = getattr(self, "speculative_config", None)
            if isinstance(raw_spec, dict):
                normalized = {k.replace("-", "_"): v for k, v in raw_spec.items()}
                if getattr(self, "spec_tokens", None) is not None and (
                    "num_speculative_tokens" not in normalized
                ):
                    normalized["num_speculative_tokens"] = self.spec_tokens
                    self.spec_tokens = None
                resolved_ckpt = _maybe_rewrite_kvn_speculative_dict(normalized)
                self.speculative_config = normalized

            spec_obj = orig_create_spec(self, *args, **kwargs)
            if spec_obj is not None and resolved_ckpt is not None:
                object.__setattr__(spec_obj, "kvn_checkpoint_path", resolved_ckpt)
            return spec_obj

        engine_args_cls.create_speculative_config = _wrapped_create_speculative_config
    _PATCHED_CLASSES.add(engine_args_cls)


def _patch_speculative_config_and_engine_args() -> None:
    """Patch ``SpeculativeConfig`` and ``EngineArgs.create_speculative_config``."""
    spec_mod = importlib.import_module("vllm.config.speculative")
    spec_cls = getattr(spec_mod, "SpeculativeConfig", None)
    if spec_cls is not None:
        _patch_speculative_config_cls(spec_cls)

    arg_mod = importlib.import_module("vllm.engine.arg_utils")
    engine_args_cls = getattr(arg_mod, "EngineArgs", None)
    if engine_args_cls is not None:
        _patch_engine_args_cls(engine_args_cls)


def _is_kvn_proposer(drafter: Any) -> bool:
    """Check whether ``drafter`` is an instance of ``KvNDSparkVllmProposer``."""
    if drafter is None:
        return False
    cls = type(drafter)
    return cls.__name__ == "KvNDSparkVllmProposer" and "kvn_proposer" in getattr(
        cls, "__module__", ""
    )


def _patch_runner_aux_layers_hook(runner_cls: type[Any]) -> None:
    """Preserve KvN ``aux_hidden_state_layer_ids`` before CUDA Graph capture."""
    attr_name = "_get_eagle3_aux_layers_from_config"
    orig_get_aux = getattr(runner_cls, attr_name, None)
    if not callable(orig_get_aux):
        return

    @functools.wraps(orig_get_aux)
    def _wrapped_get_eagle3_aux_layers(self: Any) -> tuple[int, ...] | None:
        drafter = getattr(self, "drafter", None)
        if _is_kvn_proposer(drafter):
            if getattr(drafter, "drafter_model", None) is None:
                drafter.load_model()
            aux_ids = drafter.aux_hidden_state_layer_ids
            if aux_ids:
                return tuple(aux_ids)
        return orig_get_aux(self)

    setattr(runner_cls, attr_name, _wrapped_get_eagle3_aux_layers)


def _patch_runner_propose_hook(runner_cls: type[Any]) -> None:
    """Route ``GPUModelRunner.propose_draft_token_ids`` to ``KvNDSparkVllmProposer``."""
    orig_propose = getattr(runner_cls, "propose_draft_token_ids", None)
    if not callable(orig_propose):
        return

    @functools.wraps(orig_propose)
    def _wrapped_propose(
        self: Any,
        scheduler_output: Any,
        sampled_token_ids: Any,
        sampling_metadata: Any,
        hidden_states: Any,
        sample_hidden_states: Any,
        aux_hidden_states: Any,
        spec_decode_metadata: Any,
        common_attn_metadata: Any,
        slot_mappings: Any = None,
    ) -> Any:
        drafter = getattr(self, "drafter", None)
        if _is_kvn_proposer(drafter):
            return drafter.propose_vllm_step(
                runner=self,
                scheduler_output=scheduler_output,
                sampled_token_ids=sampled_token_ids,
                aux_hidden_states=aux_hidden_states,
            )
        return orig_propose(
            self,
            scheduler_output,
            sampled_token_ids,
            sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            common_attn_metadata,
            slot_mappings,
        )

    runner_cls.propose_draft_token_ids = _wrapped_propose


def _patch_gpu_model_runner() -> None:
    """Hook ``GPUModelRunner`` when ``drafter`` is ``KvNDSparkVllmProposer``."""
    runner_mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    runner_cls = getattr(runner_mod, "GPUModelRunner", None)
    if runner_cls is None or runner_cls in _PATCHED_CLASSES:
        return

    orig_init = runner_cls.__init__
    orig_load_model = runner_cls.load_model

    @functools.wraps(orig_init)
    def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        if _is_kvn_proposer(getattr(self, "drafter", None)):
            self.use_aux_hidden_state_outputs = True

    @functools.wraps(orig_load_model)
    def _wrapped_load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = orig_load_model(self, *args, **kwargs)
        drafter = getattr(self, "drafter", None)
        if _is_kvn_proposer(drafter):
            target = (
                self.get_model()
                if hasattr(self, "get_model")
                else getattr(self, "model", None)
            )
            drafter.load_model(target_model=target)
        return result

    runner_cls.__init__ = _wrapped_init  # type: ignore[method-assign]
    runner_cls.load_model = _wrapped_load_model
    _patch_runner_aux_layers_hook(runner_cls)
    _patch_runner_propose_hook(runner_cls)
    _PATCHED_CLASSES.add(runner_cls)


def register_kvn_vllm_plugin() -> bool:
    """Entrypoint invoked by ``vllm.plugins.load_general_plugins()``.

    Returns:
        ``True`` if ``vllm`` is installed and hooks registered; ``False`` otherwise.
    """
    global _PLUGIN_REGISTERED  # noqa: PLW0603
    if _PLUGIN_REGISTERED:
        return True
    if importlib.util.find_spec("vllm") is None:
        return False

    for patch_fn, label in (
        (_patch_speculators_algos_and_base, "speculators.algos/base"),
        (_patch_speculative_config_and_engine_args, "SpeculativeConfig/EngineArgs"),
        (_patch_gpu_model_runner, "GPUModelRunner"),
    ):
        try:
            patch_fn()
        except (ImportError, AttributeError, RuntimeError, TypeError):
            logger.debug("Skipping vLLM %s patch.", label, exc_info=True)

    _PLUGIN_REGISTERED = True
    return True

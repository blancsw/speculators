"""Production vLLM V1 integration for Zero-KV-Cache ``KvN-DSpark`` drafting."""

from speculators.vllm_integration.amdahl_scheduler import (
    HardwareRooflineProfile,
    PurityLoadAwareAmdahlScheduler,
    ScheduleDecision,
    TreeScheduleDecision,
)
from speculators.vllm_integration.apc_state_cache import (
    APCBlockSnapshot,
    KvNStateSlotPool,
    PreemptedRequestState,
    ZeroReplayAPCStateCache,
    compute_chained_block_hashes,
)
from speculators.vllm_integration.kvn_proposer import (
    KvNDSparkVllmProposer,
    KvNProposeOutput,
    SpecInferStepResult,
    build_tree_attention_mask,
    compact_tree_kv_cache,
    specinfer_verify_tree_step,
)
from speculators.vllm_integration.plugin import (
    PROPOSER_IMPORT_PATH,
    configure_kvn_speculator_dict,
    is_kvn_vllm_plugin_registered,
    register_kvn_vllm_plugin,
    update_kvn_dspark_pretrained_config,
)

__all__ = [
    "PROPOSER_IMPORT_PATH",
    "APCBlockSnapshot",
    "HardwareRooflineProfile",
    "KvNDSparkVllmProposer",
    "KvNProposeOutput",
    "KvNStateSlotPool",
    "PreemptedRequestState",
    "PurityLoadAwareAmdahlScheduler",
    "ScheduleDecision",
    "SpecInferStepResult",
    "TreeScheduleDecision",
    "ZeroReplayAPCStateCache",
    "build_tree_attention_mask",
    "compact_tree_kv_cache",
    "compute_chained_block_hashes",
    "configure_kvn_speculator_dict",
    "is_kvn_vllm_plugin_registered",
    "register_kvn_vllm_plugin",
    "specinfer_verify_tree_step",
    "update_kvn_dspark_pretrained_config",
]

"""Slot-indexed persistent FP32 KvN state pool and Zero-Replay APC prefix cache.

Implements Axis 3 of ``KvN-DSpark v3`` (``deepseekv4.1.pdf`` Sec. 3.2.2 vs Markovian
Bosonic SSM state checkpointing) and resolves audit finding ``[M5]`` / ``[C6]``:

1. ``KvNStateSlotPool`` manages persistent per-sequence GPU buffers:
   - ``q_pool``, ``p_pool`` (and optional Lindblad thermal excess variance ``nu_pool``)
     in ``float32`` (``[max_num_seqs, num_layers, n_modes]``) so recurrent decoding
     never suffers ``bfloat16`` phase/amplitude truncation drift across 8k+ tokens.
   - ``last_fc_pool`` (``[max_num_seqs, d_model]``) + ``has_last_fc`` flags so
     **vLLM V1 chunked prefill** and request preemption/re-scheduling preserve the
     exact causal shift ``x_t = fc_{t-1} + embed(y_t)`` across arbitrary chunk splits
     with **0.0** boundary state corruption.
2. ``ZeroReplayAPCStateCache`` hashes 16-token blocks (matching vLLM PagedAttention
   ``block_size=16``) via collision-resistant chained ``blake2b`` digests, snapshots
   ``(q, p, [nu], last_fc)`` at full-block boundaries (``320 B/tok`` pure FP32
   ``(q, p)`` at ``L=5, M=128``; ``480 B/tok`` with ``nu``), and restores exact
   recurrent state in **0 model FLOPs** on prefix cache hits (eliminating
   DeepSeek-V4.1-Flash's 128-token SWA Bounded Replay overhead).
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

_RANK_2D = 2
_NU_MIN_COMPONENTS = 3

__all__ = [
    "APCBlockSnapshot",
    "KvNStateSlotPool",
    "PreemptedRequestState",
    "ZeroReplayAPCStateCache",
    "compute_chained_block_hashes",
]


@dataclass(frozen=True)
class APCBlockSnapshot:
    """Immutable 16-token block boundary state snapshot for Zero-Replay APC."""

    block_hash: int
    parent_hash: int
    token_end_offset: int
    q_state: Tensor  # [num_layers, n_modes]
    p_state: Tensor  # [num_layers, n_modes]
    last_fc: Tensor  # [d_model]
    nu_state: Tensor | None = None  # [num_layers, n_modes]

    def recurrent_bytes(self) -> int:
        """Return byte size of recurrent Bosonic state (q, p, [nu]) in this snapshot."""
        total = self.q_state.numel() * self.q_state.element_size()
        total += self.p_state.numel() * self.p_state.element_size()
        if self.nu_state is not None:
            total += self.nu_state.numel() * self.nu_state.element_size()
        return total

    def total_bytes(self) -> int:
        """Return total byte size including the chunk-boundary ``last_fc`` vector."""
        return self.recurrent_bytes() + (
            self.last_fc.numel() * self.last_fc.element_size()
        )


@dataclass
class PreemptedRequestState:
    """Host-offloaded exact request state across vLLM V1 scheduler preemptions."""

    req_id: str
    seq_len: int
    has_last_fc: bool
    q_state: Tensor
    p_state: Tensor
    last_fc: Tensor
    nu_state: Tensor | None = None


def _hash_token_block(
    parent_hash: int,
    block_tokens: Sequence[int],
    namespace: int = 0,
) -> int:
    """Compute a deterministic 64-bit chained digest for one ``block_size`` chunk."""
    hasher = hashlib.blake2b(digest_size=8)
    header = struct.pack(
        "<QQI", parent_hash & ((1 << 64) - 1), namespace, len(block_tokens)
    )
    hasher.update(header)
    hasher.update(
        struct.pack(f"<{len(block_tokens)}q", *(int(t) for t in block_tokens))
    )
    return int.from_bytes(hasher.digest(), byteorder="little", signed=False)


def compute_chained_block_hashes(
    token_ids: Sequence[int] | Tensor,
    block_size: int = 16,
    parent_hash: int = 0,
    namespace: int = 0,
) -> list[int]:
    """Compute chained prefix block hashes for all full ``block_size`` blocks."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    toks = (
        token_ids.detach().cpu().tolist()
        if isinstance(token_ids, Tensor)
        else [int(t) for t in token_ids]
    )
    num_full_blocks = len(toks) // block_size
    hashes: list[int] = []
    cur_hash = parent_hash
    for b_idx in range(num_full_blocks):
        start = b_idx * block_size
        chunk = toks[start : start + block_size]
        cur_hash = _hash_token_block(cur_hash, chunk, namespace=namespace)
        hashes.append(cur_hash)
    return hashes


class KvNStateSlotPool:
    """Slot-indexed persistent GPU state manager for ``KvN-DSpark`` in vLLM V1.

    Maintains persistent ``float32`` recurrent state ``(q, p, [nu])`` and chunk-boundary
    ``last_fc`` across ``max_num_seqs`` active slots, supporting:
    - Exact multi-chunk prefill continuation (zero boundary state loss);
    - Host swap-out / resume on scheduler preemption;
    - Batched gather/scatter to ``KvNDSparkDraftModel`` ``state_cache`` tensors.
    """

    def __init__(
        self,
        max_num_seqs: int,
        num_layers: int,
        n_modes: int,
        d_model: int,
        use_lindblad_purity: bool = False,
        state_dtype: torch.dtype = torch.float32,
        fc_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> None:
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be > 0, got {max_num_seqs}.")
        if num_layers <= 0 or n_modes <= 0 or d_model <= 0:
            raise ValueError(
                "num_layers, n_modes, and d_model must all be strictly positive."
            )
        self.max_num_seqs = int(max_num_seqs)
        self.num_layers = int(num_layers)
        self.n_modes = int(n_modes)
        self.d_model = int(d_model)
        self.use_lindblad_purity = bool(use_lindblad_purity)
        self.state_dtype = state_dtype
        self.fc_dtype = fc_dtype
        self.device = torch.device(device)

        shape = (self.max_num_seqs, self.num_layers, self.n_modes)
        self.q_pool = torch.zeros(shape, dtype=self.state_dtype, device=self.device)
        self.p_pool = torch.zeros(shape, dtype=self.state_dtype, device=self.device)
        self.nu_pool: Tensor | None = (
            torch.zeros(shape, dtype=self.state_dtype, device=self.device)
            if self.use_lindblad_purity
            else None
        )
        self.last_fc_pool = torch.zeros(
            (self.max_num_seqs, self.d_model),
            dtype=self.fc_dtype,
            device=self.device,
        )
        self.has_last_fc = torch.zeros(
            (self.max_num_seqs,), dtype=torch.bool, device=self.device
        )
        self.seq_lens = torch.zeros(
            (self.max_num_seqs,), dtype=torch.int64, device=self.device
        )

        self._req_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self.max_num_seqs - 1, -1, -1))
        self._preempted: dict[str, PreemptedRequestState] = {}

    @property
    def num_active_requests(self) -> int:
        """Number of currently allocated sequence slots."""
        return len(self._req_to_slot)

    def active_request_ids(self) -> list[str]:
        """Return list of currently active ``req_id`` keys owning GPU slots."""
        return list(self._req_to_slot.keys())

    def has_request(self, req_id: str) -> bool:
        """Return whether ``req_id`` currently owns an active GPU slot."""
        return req_id in self._req_to_slot

    def is_preempted(self, req_id: str) -> bool:
        """Return whether ``req_id`` has a saved preempted state waiting on CPU."""
        return req_id in self._preempted

    @property
    def _preempted_snapshots(self) -> dict[str, PreemptedRequestState]:
        """Compatibility alias exposing CPU-offloaded preempted snapshots."""
        return self._preempted

    def get_slot(self, req_id: str) -> int:
        """Return the slot index assigned to ``req_id``."""
        if req_id not in self._req_to_slot:
            raise KeyError(f"Request {req_id!r} is not allocated in KvNStateSlotPool.")
        return self._req_to_slot[req_id]

    def allocate_slot(self, req_id: str) -> int:
        """Allocate or restore a slot for ``req_id`` and return ``slot_idx``."""
        slot_idx, _ = self.get_or_allocate_slot(req_id)
        return slot_idx

    def resolve_slot_or_restore(self, req_id: str) -> int:
        """Return active slot for ``req_id``, auto-restoring if preempted."""
        if req_id in self._req_to_slot:
            return self._req_to_slot[req_id]
        if req_id in self._preempted:
            slot_idx, _ = self.get_or_allocate_slot(req_id)
            return slot_idx
        raise KeyError(f"Request {req_id!r} is not allocated in KvNStateSlotPool.")

    def reset_slot(self, slot_idx: int) -> None:
        """Zero all recurrent and boundary state for ``slot_idx``."""
        self.q_pool[slot_idx].zero_()
        self.p_pool[slot_idx].zero_()
        if self.nu_pool is not None:
            self.nu_pool[slot_idx].zero_()
        self.last_fc_pool[slot_idx].zero_()
        self.has_last_fc[slot_idx] = False
        self.seq_lens[slot_idx] = 0

    def get_or_allocate_slot(self, req_id: str) -> tuple[int, bool]:
        """Return ``(slot_idx, is_new_allocation)``, auto-restoring preempted state."""
        if req_id in self._req_to_slot:
            return self._req_to_slot[req_id], False
        if not self._free_slots:
            raise RuntimeError(
                f"KvNStateSlotPool exhausted (max_num_seqs={self.max_num_seqs})."
            )
        slot_idx = self._free_slots.pop()
        self._req_to_slot[req_id] = slot_idx
        if req_id in self._preempted:
            saved = self._preempted.pop(req_id)
            self.q_pool[slot_idx].copy_(saved.q_state.to(self.device, self.state_dtype))
            self.p_pool[slot_idx].copy_(saved.p_state.to(self.device, self.state_dtype))
            if self.nu_pool is not None and saved.nu_state is not None:
                self.nu_pool[slot_idx].copy_(
                    saved.nu_state.to(self.device, self.state_dtype)
                )
            elif self.nu_pool is not None:
                self.nu_pool[slot_idx].zero_()
            self.last_fc_pool[slot_idx].copy_(
                saved.last_fc.to(self.device, self.fc_dtype)
            )
            self.has_last_fc[slot_idx] = saved.has_last_fc
            self.seq_lens[slot_idx] = saved.seq_len
            return slot_idx, False

        self.reset_slot(slot_idx)
        return slot_idx, True

    def preempt_request(self, req_id: str) -> None:
        """Swap out ``req_id``'s exact FP32 recurrent + boundary state to CPU RAM."""
        if req_id not in self._req_to_slot:
            return
        slot_idx = self._req_to_slot.pop(req_id)
        self._preempted[req_id] = PreemptedRequestState(
            req_id=req_id,
            seq_len=int(self.seq_lens[slot_idx].item()),
            has_last_fc=bool(self.has_last_fc[slot_idx].item()),
            q_state=self.q_pool[slot_idx].detach().cpu().clone(),
            p_state=self.p_pool[slot_idx].detach().cpu().clone(),
            last_fc=self.last_fc_pool[slot_idx].detach().cpu().clone(),
            nu_state=(
                self.nu_pool[slot_idx].detach().cpu().clone()
                if self.nu_pool is not None
                else None
            ),
        )
        self.reset_slot(slot_idx)
        self._free_slots.append(slot_idx)

    def free_request(self, req_id: str) -> None:
        """Permanently release ``req_id`` from active and preempted tables."""
        self._preempted.pop(req_id, None)
        if req_id in self._req_to_slot:
            slot_idx = self._req_to_slot.pop(req_id)
            self.reset_slot(slot_idx)
            self._free_slots.append(slot_idx)

    def build_causal_shifted_fc(
        self,
        slot_idx: int,
        chunk_fc: Tensor,
    ) -> Tensor:
        """Build ``shifted_fc`` ``[chunk_len, d_model]`` for chunked prefill/decode.

        - If ``has_last_fc[slot_idx]`` is ``False`` (chunk 0 at sequence start):
          ``shifted_fc[0] = 0``, ``shifted_fc[1:] = chunk_fc[:-1]``.
        - If ``has_last_fc[slot_idx]`` is ``True`` (continuation chunk ``c > 0``):
          ``shifted_fc[0] = last_fc_pool[slot_idx]``,
          ``shifted_fc[1:] = chunk_fc[:-1]``.
        """
        if chunk_fc.ndim != _RANK_2D or chunk_fc.size(1) != self.d_model:
            raise ValueError(
                f"Expected chunk_fc [T, {self.d_model}], got {tuple(chunk_fc.shape)}."
            )
        chunk_len = chunk_fc.size(0)
        if chunk_len == 0:
            return chunk_fc.clone()
        first_fc = (
            self.last_fc_pool[slot_idx].to(chunk_fc.dtype).unsqueeze(0)
            if bool(self.has_last_fc[slot_idx].item())
            else chunk_fc.new_zeros((1, self.d_model))
        )
        if chunk_len == 1:
            return first_fc
        return torch.cat([first_fc, chunk_fc[:-1]], dim=0)

    def commit_slot_state(
        self,
        slot_idx: int,
        q_state: Tensor,
        p_state: Tensor,
        last_fc: Tensor,
        tokens_advanced: int,
        nu_state: Tensor | None = None,
    ) -> None:
        """Commit updated recurrent state and boundary ``last_fc`` into ``slot_idx``."""
        self.q_pool[slot_idx].copy_(q_state.to(self.device, self.state_dtype))
        self.p_pool[slot_idx].copy_(p_state.to(self.device, self.state_dtype))
        if self.nu_pool is not None:
            if nu_state is not None:
                self.nu_pool[slot_idx].copy_(nu_state.to(self.device, self.state_dtype))
            else:
                self.nu_pool[slot_idx].zero_()
        self.last_fc_pool[slot_idx].copy_(
            last_fc.reshape(self.d_model).to(self.device, self.fc_dtype)
        )
        self.has_last_fc[slot_idx] = True
        self.seq_lens[slot_idx] += int(tokens_advanced)

    def gather_drafter_state_cache(
        self,
        slot_indices: Sequence[int] | Tensor,
        include_nu: bool | None = None,
    ) -> Tensor:
        """Gather active slots into ``[num_layers, C, batch_size, n_modes]`` tensor.

        ``C = 3`` when ``include_nu`` is ``True`` and ``nu_pool`` is active, else ``2``.
        """
        idx = (
            slot_indices.to(device=self.device, dtype=torch.long)
            if isinstance(slot_indices, Tensor)
            else torch.tensor(slot_indices, dtype=torch.long, device=self.device)
        )
        # q_b, p_b: [B, L, M] -> permute to [L, B, M]
        q_b = self.q_pool.index_select(0, idx).permute(1, 0, 2)
        p_b = self.p_pool.index_select(0, idx).permute(1, 0, 2)
        want_nu = self.use_lindblad_purity if include_nu is None else bool(include_nu)
        if want_nu and self.nu_pool is not None:
            nu_b = self.nu_pool.index_select(0, idx).permute(1, 0, 2)
            return torch.stack([q_b, p_b, nu_b], dim=1).contiguous()
        return torch.stack([q_b, p_b], dim=1).contiguous()

    def scatter_drafter_state_cache(
        self,
        slot_indices: Sequence[int] | Tensor,
        state_cache: Tensor | tuple[Tensor, ...] | dict[str, Tensor],
    ) -> None:
        """Write updated ``state_cache`` back into the persistent slot pool."""
        idx = (
            slot_indices.to(device=self.device, dtype=torch.long)
            if isinstance(slot_indices, Tensor)
            else torch.tensor(slot_indices, dtype=torch.long, device=self.device)
        )
        if isinstance(state_cache, dict):
            q_new = state_cache["q"].permute(1, 0, 2).to(self.device, self.state_dtype)
            p_new = state_cache["p"].permute(1, 0, 2).to(self.device, self.state_dtype)
            nu_raw = state_cache.get("nu")
            nu_new = (
                nu_raw.permute(1, 0, 2).to(self.device, self.state_dtype)
                if nu_raw is not None
                else None
            )
        elif isinstance(state_cache, tuple):
            q_new = state_cache[0].permute(1, 0, 2).to(self.device, self.state_dtype)
            p_new = state_cache[1].permute(1, 0, 2).to(self.device, self.state_dtype)
            nu_new = (
                state_cache[2].permute(1, 0, 2).to(self.device, self.state_dtype)
                if len(state_cache) >= _NU_MIN_COMPONENTS and state_cache[2] is not None
                else None
            )
        else:
            q_new = (
                state_cache[:, 0, :, :]
                .permute(1, 0, 2)
                .to(self.device, self.state_dtype)
            )
            p_new = (
                state_cache[:, 1, :, :]
                .permute(1, 0, 2)
                .to(self.device, self.state_dtype)
            )
            nu_new = (
                state_cache[:, 2, :, :]
                .permute(1, 0, 2)
                .to(self.device, self.state_dtype)
                if state_cache.size(1) >= _NU_MIN_COMPONENTS
                else None
            )

        self.q_pool.index_copy_(0, idx, q_new)
        self.p_pool.index_copy_(0, idx, p_new)
        if self.nu_pool is not None and nu_new is not None:
            self.nu_pool.index_copy_(0, idx, nu_new)

    def bytes_per_seq(self, include_last_fc: bool = True) -> int:
        """Compute exact persistent GPU state bytes per active sequence."""
        state_elem = torch.empty((), dtype=self.state_dtype).element_size()
        n_comp = 3 if self.use_lindblad_purity else 2
        rec_bytes = self.num_layers * n_comp * self.n_modes * state_elem
        if not include_last_fc:
            return rec_bytes
        fc_elem = torch.empty((), dtype=self.fc_dtype).element_size()
        return rec_bytes + self.d_model * fc_elem


class ZeroReplayAPCStateCache:
    """Zero-Replay Automatic Prefix Cache (APC) for Markovian ``KvN-DSpark`` states.

    Snapshots ``(q, p, [nu], last_fc)`` at full ``block_size=16`` token boundaries
    keyed by chained 64-bit ``blake2b`` block hashes. On any multi-turn or shared
    system-prompt prefix hit, restores the exact FP32 recurrent state into
    ``KvNStateSlotPool`` in **0 model FLOPs** (zero 128-token SWA Bounded Replay).
    """

    def __init__(
        self,
        block_size: int = 16,
        max_cached_blocks: int = 4096,
        snapshot_device: torch.device | str = "cpu",
        namespace: int = 0,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}.")
        if max_cached_blocks <= 0:
            raise ValueError(f"max_cached_blocks must be > 0, got {max_cached_blocks}.")
        self.block_size = int(block_size)
        self.max_cached_blocks = int(max_cached_blocks)
        self.snapshot_device = torch.device(snapshot_device)
        self.namespace = int(namespace)

        self._blocks: OrderedDict[int, APCBlockSnapshot] = OrderedDict()
        self.num_lookups = 0
        self.num_block_hits = 0
        self.num_request_hits = 0
        self.saved_prefill_tokens = 0
        self.num_evictions = 0

    def __len__(self) -> int:
        return len(self._blocks)

    def contains_block(self, block_hash: int) -> bool:
        """Return whether ``block_hash`` is currently resident in the APC table."""
        return block_hash in self._blocks

    def store_block_snapshot(
        self,
        block_hash: int,
        parent_hash: int,
        token_end_offset: int,
        q_state: Tensor,
        p_state: Tensor,
        last_fc: Tensor,
        nu_state: Tensor | None = None,
    ) -> APCBlockSnapshot:
        """Insert or refresh an LRU 16-token boundary snapshot."""
        if block_hash in self._blocks:
            self._blocks.move_to_end(block_hash)
            return self._blocks[block_hash]

        while len(self._blocks) >= self.max_cached_blocks:
            self._blocks.popitem(last=False)
            self.num_evictions += 1

        snap = APCBlockSnapshot(
            block_hash=block_hash,
            parent_hash=parent_hash,
            token_end_offset=int(token_end_offset),
            q_state=q_state.detach().to(self.snapshot_device).clone(),
            p_state=p_state.detach().to(self.snapshot_device).clone(),
            last_fc=last_fc.detach().reshape(-1).to(self.snapshot_device).clone(),
            nu_state=(
                nu_state.detach().to(self.snapshot_device).clone()
                if nu_state is not None
                else None
            ),
        )
        self._blocks[block_hash] = snap
        return snap

    def record_sequence_blocks(
        self,
        token_ids: Sequence[int] | Tensor,
        q_per_step: Tensor,  # [num_layers, seq_len, n_modes]
        p_per_step: Tensor,  # [num_layers, seq_len, n_modes]
        fc_per_step: Tensor,  # [seq_len, d_model]
        nu_per_step: Tensor | None = None,  # [num_layers, seq_len, n_modes]
        start_offset: int = 0,
        initial_parent_hash: int = 0,
    ) -> list[int]:
        """Snapshot every full ``block_size`` boundary crossed by ``token_ids``.

        ``token_ids`` may be a full prompt (``start_offset=0``) or an aligned chunk
        beginning at a multiple of ``block_size`` with ``initial_parent_hash`` set to
        the previous block's chained hash.
        """
        if start_offset % self.block_size != 0:
            raise ValueError(
                f"start_offset ({start_offset}) must be divisible by "
                f"block_size ({self.block_size})."
            )
        toks = (
            token_ids.detach().cpu().tolist()
            if isinstance(token_ids, Tensor)
            else [int(t) for t in token_ids]
        )
        hashes = compute_chained_block_hashes(
            toks,
            block_size=self.block_size,
            parent_hash=initial_parent_hash,
            namespace=self.namespace,
        )
        parent = initial_parent_hash
        for b_idx, b_hash in enumerate(hashes):
            local_end_idx = (b_idx + 1) * self.block_size - 1
            global_end = start_offset + local_end_idx + 1
            self.store_block_snapshot(
                block_hash=b_hash,
                parent_hash=parent,
                token_end_offset=global_end,
                q_state=q_per_step[:, local_end_idx, :],
                p_state=p_per_step[:, local_end_idx, :],
                last_fc=fc_per_step[local_end_idx, :],
                nu_state=(
                    nu_per_step[:, local_end_idx, :]
                    if nu_per_step is not None
                    else None
                ),
            )
            parent = b_hash
        return hashes

    def match_and_restore_prefix(
        self,
        token_ids: Sequence[int] | Tensor,
        slot_pool: KvNStateSlotPool,
        slot_idx: int,
        require_uncomputed_tail: bool = True,
    ) -> tuple[int, int, list[int]]:
        """Restore longest cached prefix into ``slot_pool[slot_idx]`` in 0 FLOPs.

        Args:
            token_ids: Full prompt token sequence of length ``P``.
            slot_pool: Target ``KvNStateSlotPool`` to populate.
            slot_idx: Target slot index in ``slot_pool``.
            require_uncomputed_tail: When ``True`` (vLLM APC convention), leaves at
                least 1 final prompt token uncached when ``P`` is an exact multiple of
                ``block_size`` so the verifier computes next-token logits on the tail.

        Returns:
            ``(restored_token_count, last_hit_block_hash, uncached_suffix_tokens)``.
        """
        toks = (
            token_ids.detach().cpu().tolist()
            if isinstance(token_ids, Tensor)
            else [int(t) for t in token_ids]
        )
        self.num_lookups += 1
        if not toks:
            slot_pool.reset_slot(slot_idx)
            return 0, 0, []

        effective_len = len(toks) - 1 if require_uncomputed_tail else len(toks)
        if effective_len < self.block_size:
            slot_pool.reset_slot(slot_idx)
            return 0, 0, toks

        hashes = compute_chained_block_hashes(
            toks[:effective_len],
            block_size=self.block_size,
            parent_hash=0,
            namespace=self.namespace,
        )
        last_snap: APCBlockSnapshot | None = None
        hit_blocks = 0
        for b_hash in hashes:
            snap = self._blocks.get(b_hash)
            if snap is None:
                break
            self._blocks.move_to_end(b_hash)
            last_snap = snap
            hit_blocks += 1

        if last_snap is None or hit_blocks == 0:
            slot_pool.reset_slot(slot_idx)
            return 0, 0, toks

        restored_tokens = hit_blocks * self.block_size
        slot_pool.reset_slot(slot_idx)
        slot_pool.commit_slot_state(
            slot_idx=slot_idx,
            q_state=last_snap.q_state,
            p_state=last_snap.p_state,
            last_fc=last_snap.last_fc,
            tokens_advanced=restored_tokens,
            nu_state=last_snap.nu_state,
        )
        self.num_block_hits += hit_blocks
        self.num_request_hits += 1
        self.saved_prefill_tokens += restored_tokens
        return restored_tokens, last_snap.block_hash, toks[restored_tokens:]

    def memory_stats(self) -> dict[str, Any]:
        """Return exact byte-accounting and zero-replay hit statistics."""
        rec_bytes = sum(s.recurrent_bytes() for s in self._blocks.values())
        tot_bytes = sum(s.total_bytes() for s in self._blocks.values())
        n_blocks = len(self._blocks)
        toks_represented = n_blocks * self.block_size
        return {
            "block_size": self.block_size,
            "cached_blocks": n_blocks,
            "max_cached_blocks": self.max_cached_blocks,
            "recurrent_bytes_total": rec_bytes,
            "total_bytes_with_last_fc": tot_bytes,
            "recurrent_bytes_per_token": (
                round(rec_bytes / toks_represented, 2) if toks_represented > 0 else 0.0
            ),
            "total_bytes_per_token": (
                round(tot_bytes / toks_represented, 2) if toks_represented > 0 else 0.0
            ),
            "replay_flops_per_hit": 0,
            "num_lookups": self.num_lookups,
            "num_request_hits": self.num_request_hits,
            "num_block_hits": self.num_block_hits,
            "saved_prefill_tokens": self.saved_prefill_tokens,
            "num_evictions": self.num_evictions,
        }

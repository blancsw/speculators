"""KvN-DSpark optimizer builders: BlockwiseMuon and SinkhornAdamW."""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.nn import Module
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

from speculators.train.optimizers import split_named_params_for_muon

__all__ = [
    "BlockwiseMuon",
    "SinkhornAdamW",
    "build_kvn_optimizers",
    "infer_muon_block_partition",
    "should_apply_weight_decay",
    "sinkhorn_normalize_2d",
    "split_adamw_param_groups",
]

logger = logging.getLogger("speculators")

_MATRIX_NDIM = 2

_NO_WEIGHT_DECAY_NAME_HINTS = (
    "bias",
    "norm",
    "omega_rff",
    "phase_rff",
    "log_eta",
    "kvn_omega",
    "dyad_",
    "vacuum_",
    "horizon_",
    "squeezing",
    "butterfly",
    "lindblad",
    "layer_mode_",
    "engram_table",
)


def should_apply_weight_decay(name: str, param: Tensor) -> bool:
    """Return False for 1D/scalar tensors, norms, biases, and spectral params."""
    if param.ndim < _MATRIX_NDIM or min(param.shape) <= 1:
        return False
    return not any(hint in name for hint in _NO_WEIGHT_DECAY_NAME_HINTS)


@torch.no_grad()
def sinkhorn_normalize_2d(
    mat: Tensor,
    num_iters: int = 5,
    eps: float = 1e-8,
) -> Tensor:
    """5-iteration Sinkhorn RMS equilibration over active 2D rows and columns."""
    if mat.ndim != _MATRIX_NDIM or mat.numel() == 0:
        return mat
    x = mat.float()
    orig_norm = x.norm()
    if orig_norm <= eps:
        return mat
    active_rows = x.abs().amax(dim=-1) > eps
    if not bool(active_rows.any()):
        return mat
    x_sub = x[active_rows]
    for _ in range(num_iters):
        row_rms = (x_sub.square().mean(dim=-1, keepdim=True) + eps).sqrt()
        x_sub = x_sub / row_rms
        col_rms = (x_sub.square().mean(dim=0, keepdim=True) + eps).sqrt()
        x_sub = x_sub / col_rms
    scale = orig_norm / x_sub.norm().clamp_min(eps)
    out = torch.zeros_like(x)
    out[active_rows] = x_sub * scale
    return out.to(dtype=mat.dtype)


def infer_muon_block_partition(name: str, shape: torch.Size) -> tuple[int, int] | None:
    """Infer ``(split_dim, num_blocks)`` for fused QKV or multi-layer ``fc`` weights."""
    if len(shape) != _MATRIX_NDIM:
        return None
    rows, cols = int(shape[0]), int(shape[1])
    if ("qkv" in name) and rows == 3 * cols and cols > 1:
        return (0, 3)
    if (
        (name == "fc.weight" or name.endswith(".fc.weight"))
        and cols > rows > 1
        and cols % rows == 0
    ):
        return (1, cols // rows)
    return None


class SinkhornAdamW(torch.optim.AdamW):
    """``torch.optim.AdamW`` with 5-step Sinkhorn equilibration on ``engram`` groups."""

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        for group in self.param_groups:
            if not group.get("sinkhorn_engram", False):
                continue
            n_iters = int(group.get("sinkhorn_iters", 5))
            for p in group["params"]:
                if (
                    p.grad is not None
                    and not p.grad.is_sparse
                    and not isinstance(p.grad, DTensor)
                    and p.grad.ndim == _MATRIX_NDIM
                ):
                    p.grad.copy_(sinkhorn_normalize_2d(p.grad, num_iters=n_iters))
        return super().step(closure)


class BlockwiseMuon(torch.optim.Muon):
    """``torch.optim.Muon`` orthogonalizing stacked QKV/``fc`` blocks independently."""

    def __init__(
        self,
        params,
        *,
        block_partitions: dict[int, tuple[int, int]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self._block_partitions: dict[int, tuple[int, int]] = dict(
            block_partitions or {}
        )

    def _update_single_param(self, p: Tensor, group: dict) -> None:
        if p.grad is None:
            return
        grad = p.grad
        lr = float(group["lr"])
        momentum = float(group["momentum"])
        ns_coefficients = group["ns_coefficients"]
        eps = float(group["eps"])
        ns_steps = int(group["ns_steps"])
        adjust_lr_fn = group["adjust_lr_fn"]

        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                grad, memory_format=torch.preserve_format
            )
        buf: Tensor = state["momentum_buffer"]
        buf.lerp_(grad, 1.0 - momentum)
        raw_update = grad.lerp(buf, momentum) if group["nesterov"] else buf

        part = self._block_partitions.get(id(p))
        if part is not None:
            split_dim, n_blocks = part
            chunks = raw_update.chunk(n_blocks, dim=split_dim)
            ortho_chunks = [
                _zeropower_via_newtonschulz(
                    c.contiguous(), ns_coefficients, ns_steps, eps
                )
                for c in chunks
            ]
            ortho_update = torch.cat(ortho_chunks, dim=split_dim)
            adjusted_lr = _adjust_lr(lr, adjust_lr_fn, chunks[0].shape)
        else:
            ortho_update = _zeropower_via_newtonschulz(
                raw_update, ns_coefficients, ns_steps, eps
            )
            adjusted_lr = _adjust_lr(lr, adjust_lr_fn, p.shape)

        p.mul_(1.0 - lr * float(group["weight_decay"]))
        p.add_(ortho_update, alpha=-adjusted_lr)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if not self._block_partitions or any(
            isinstance(p, DTensor)
            for group in self.param_groups
            for p in group["params"]
        ):
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                self._update_single_param(p, group)
        return loss


def split_adamw_param_groups(
    named_params: list[tuple[str, Tensor]],
    weight_decay: float,
) -> list[dict]:
    """Partition AdamW parameters into ``decay``, ``no_decay``, and ``engram``."""
    decay_params: list[Tensor] = []
    no_decay_params: list[Tensor] = []
    engram_params: list[Tensor] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if (
            "engram" in name
            and "proj" not in name
            and "norm" not in name
            and param.ndim == _MATRIX_NDIM
            and min(param.shape) > 1
        ):
            engram_params.append(param)
        elif should_apply_weight_decay(name, param):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    groups: list[dict] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if engram_params:
        groups.append(
            {
                "params": engram_params,
                "weight_decay": 0.0,
                "sinkhorn_engram": True,
                "sinkhorn_iters": 5,
            }
        )
    return groups


def build_kvn_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build BlockwiseMuon + SinkhornAdamW optimizers for KvN-DSpark."""
    fused = config.optimizer_implementation == "fused"

    if config.optimizer == "adamw":
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        param_groups = split_adamw_param_groups(trainable, config.weight_decay)
        if not param_groups:
            raise ValueError("No trainable parameters found to optimize.")
        return [
            SinkhornAdamW(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                fused=fused,
            )
        ]

    if config.optimizer == "muon":
        muon_named, adamw_named = split_named_params_for_muon(model)
        block_partitions: dict[int, tuple[int, int]] = {}
        for name, param in muon_named:
            part = infer_muon_block_partition(name, param.shape)
            if part is not None:
                block_partitions[id(param)] = part

        adamw_groups = split_adamw_param_groups(adamw_named, config.weight_decay)
        logger.info(
            "KvN Muon: %d 2D params via BlockwiseMuon (%d block-wise), "
            "%d params via SinkhornAdamW.",
            len(muon_named),
            len(block_partitions),
            len(adamw_named),
        )
        optimizers: list[torch.optim.Optimizer] = []
        if muon_named:
            optimizers.append(
                BlockwiseMuon(
                    [p for _, p in muon_named],
                    block_partitions=block_partitions,
                    lr=config.muon_lr,
                    momentum=config.muon_momentum,
                    weight_decay=config.muon_weight_decay,
                    ns_steps=config.muon_ns_steps,
                    adjust_lr_fn=config.muon_adjust_lr_fn,
                )
            )
        if adamw_groups:
            optimizers.append(
                SinkhornAdamW(
                    adamw_groups,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    fused=fused,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")

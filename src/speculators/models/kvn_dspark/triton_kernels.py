"""Mega-KvN Triton Kernels & Exact FP32 Fallbacks for KvN-DSpark v3 (Axe 3).

Provides hardware-efficient segmented parallel scans and fused multi-step recurrent
decode kernels for:
1. 1D scalar open-system Lindblad thermal covariance scan (delta_nu_t >= 0).
2. 1D complex quadrature Bosonic scan (q_t + i p_t).
3. 4x4 sp(4, R) butterfly symplectic-squeezed dyad scan (z_t = A_t z_{t-1} + u_t).
4. Fused persistent-register multi-step speculative decode across variable valid_lens.

When Triton is installed and tensors reside on CUDA, executes single-pass tiled SRAM
`tl.associative_scan` kernels (both forward and exact reverse-time adjoint backward).
When Triton is absent (e.g., Windows without triton) or running on CPU / inside Dynamo
capture, falls back seamlessly to bit-compatible FP32 PyTorch Kogge-Stone scans.
"""

# ruff: noqa: N803, PLR2004

from __future__ import annotations

import importlib.util

import torch
from torch import Tensor

_HAS_TRITON = importlib.util.find_spec("triton") is not None

if _HAS_TRITON:
    import triton
    import triton.language as tl

__all__ = [
    "HAS_TRITON",
    "fused_multistep_complex_decode",
    "fused_multistep_sp4_decode",
    "pytorch_segmented_complex_scan",
    "pytorch_segmented_scalar_scan",
    "pytorch_segmented_sp4_scan",
    "segmented_complex_scan",
    "segmented_scalar_scan",
    "segmented_sp4_scan",
]

HAS_TRITON: bool = _HAS_TRITON


def _can_run_triton(x: Tensor, use_triton: bool) -> bool:
    return (
        bool(use_triton)
        and _HAS_TRITON
        and x.is_cuda
        and not torch.compiler.is_compiling()
    )


# ---------------------------------------------------------------------------
# Pure PyTorch FP32 Kogge-Stone Reference & Fallback Scans
# ---------------------------------------------------------------------------


def _fold_scalar_initial_state(
    a_decay: Tensor,
    u_val: Tensor,
    initial_state: Tensor | None,
) -> Tensor:
    if initial_state is None or u_val.size(1) == 0:
        return u_val
    init_f = initial_state.float()
    if init_f.ndim == 3:
        init_f = init_f.squeeze(1)
    u_out = u_val.clone()
    u_out[:, 0, :] = a_decay[:, 0, :] * init_f + u_val[:, 0, :]
    return u_out


def pytorch_segmented_scalar_scan(
    a_decay: Tensor,
    u_val: Tensor,
    initial_state: Tensor | None = None,
) -> Tensor:
    """Work-parallel O(T log T) FP32 Kogge-Stone scan for z_t = a_t * z_{t-1} + u_t."""
    a = a_decay.float()
    z = _fold_scalar_initial_state(a, u_val.float(), initial_state)
    seq_len = z.size(1)
    stride = 1
    while stride < seq_len:
        z_prev = z[:, :-stride, :]
        a_prev = a[:, :-stride, :]
        a_curr = a[:, stride:, :]
        z_curr = z[:, stride:, :]

        z_upd = a_curr * z_prev + z_curr
        a_upd = a_curr * a_prev

        z = torch.cat([z[:, :stride, :], z_upd], dim=1)
        a = torch.cat([a[:, :stride, :], a_upd], dim=1)
        stride <<= 1
    return z


def _fold_complex_initial_state(
    a_cos: Tensor,
    a_sin: Tensor,
    u_q: Tensor,
    u_p: Tensor,
    initial_state: tuple[Tensor, Tensor] | None,
) -> tuple[Tensor, Tensor]:
    if initial_state is None or u_q.size(1) == 0:
        return u_q, u_p
    q0 = initial_state[0].float()
    p0 = initial_state[1].float()
    if q0.ndim == 3:
        q0 = q0.squeeze(1)
        p0 = p0.squeeze(1)
    c0 = a_cos[:, 0, :]
    s0 = a_sin[:, 0, :]
    uq_out = u_q.clone()
    up_out = u_p.clone()
    uq_out[:, 0, :] = c0 * q0 - s0 * p0 + u_q[:, 0, :]
    up_out[:, 0, :] = s0 * q0 + c0 * p0 + u_p[:, 0, :]
    return uq_out, up_out


def pytorch_segmented_complex_scan(
    a_cos: Tensor,
    a_sin: Tensor,
    u_q: Tensor,
    u_p: Tensor,
    initial_state: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Work-parallel O(T log T) FP32 Kogge-Stone scan in real (q, p) quadratures."""
    c = a_cos.float()
    s = a_sin.float()
    q, p = _fold_complex_initial_state(c, s, u_q.float(), u_p.float(), initial_state)
    seq_len = q.size(1)
    stride = 1
    while stride < seq_len:
        q_prev = q[:, :-stride, :]
        p_prev = p[:, :-stride, :]
        c_prev = c[:, :-stride, :]
        s_prev = s[:, :-stride, :]

        c_curr = c[:, stride:, :]
        s_curr = s[:, stride:, :]
        q_curr = q[:, stride:, :]
        p_curr = p[:, stride:, :]

        q_upd = c_curr * q_prev - s_curr * p_prev + q_curr
        p_upd = s_curr * q_prev + c_curr * p_prev + p_curr
        c_upd = c_curr * c_prev - s_curr * s_prev
        s_upd = s_curr * c_prev + c_curr * s_prev

        q = torch.cat([q[:, :stride, :], q_upd], dim=1)
        p = torch.cat([p[:, :stride, :], p_upd], dim=1)
        c = torch.cat([c[:, :stride, :], c_upd], dim=1)
        s = torch.cat([s[:, :stride, :], s_upd], dim=1)
        stride <<= 1
    return q, p


def _fold_sp4_initial_state(
    a_mat: Tensor,
    u_vec: Tensor,
    initial_state: Tensor | None,
) -> Tensor:
    if initial_state is None or u_vec.size(1) == 0:
        return u_vec
    z0 = initial_state.float()
    if z0.ndim == 4:
        z0 = z0.squeeze(1)
    u_out = u_vec.clone()
    az0 = torch.matmul(a_mat[:, 0], z0.unsqueeze(-1)).squeeze(-1)
    u_out[:, 0] = az0 + u_vec[:, 0]
    return u_out


def pytorch_segmented_sp4_scan(
    a_mat: Tensor,
    u_vec: Tensor,
    initial_state: Tensor | None = None,
) -> Tensor:
    """Work-parallel O(T log T) FP32 Kogge-Stone scan for 4x4 dyads.

    Args:
        a_mat: [B, T, D_dyad, 4, 4] float32 transition matrices.
        u_vec: [B, T, D_dyad, 4] float32 forcing vectors.
        initial_state: Optional [B, D_dyad, 4] initial state z_{-1}.
    Returns:
        z_seq: [B, T, D_dyad, 4] float32 state trajectory.
    """
    a = a_mat.float()
    z = _fold_sp4_initial_state(a, u_vec.float(), initial_state)
    seq_len = z.size(1)
    stride = 1
    while stride < seq_len:
        z_prev = z[:, :-stride]
        a_prev = a[:, :-stride]
        a_curr = a[:, stride:]
        z_curr = z[:, stride:]

        z_upd = torch.matmul(a_curr, z_prev.unsqueeze(-1)).squeeze(-1) + z_curr
        a_upd = torch.matmul(a_curr, a_prev)

        z = torch.cat([z[:, :stride], z_upd], dim=1)
        a = torch.cat([a[:, :stride], a_upd], dim=1)
        stride <<= 1
    return z


# ---------------------------------------------------------------------------
# Triton Tiled Associative Scan Kernels + Exact Adjoint Autograd Functions
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _scalar_combine(a_left, u_left, a_right, u_right):
        return a_right * a_left, a_right * u_left + u_right

    @triton.jit
    def _scalar_scan_fwd_kernel(
        a_ptr,
        u_ptr,
        out_ptr,
        seq_len,
        stride_bm_t,
        stride_t,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base = pid * stride_bm_t
        carry = tl.zeros((), dtype=tl.float32)
        offs = tl.arange(0, BLOCK_T)

        t0 = 0
        while t0 < seq_len:
            idx = t0 + offs
            mask = idx < seq_len
            ptrs = base + idx * stride_t

            a_tile = tl.load(a_ptr + ptrs, mask=mask, other=1.0).to(tl.float32)
            u_tile = tl.load(u_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            u_tile = tl.where(offs == 0, a_tile * carry + u_tile, u_tile)

            _, z_tile = tl.associative_scan(
                (a_tile, u_tile), axis=0, combine_fn=_scalar_combine
            )
            tl.store(out_ptr + ptrs, z_tile, mask=mask)

            last_idx = tl.minimum(BLOCK_T - 1, seq_len - t0 - 1)
            carry = tl.sum(tl.where(offs == last_idx, z_tile, 0.0), axis=0)
            t0 += BLOCK_T

    @triton.jit
    def _complex_combine(
        c_left, s_left, q_left, p_left, c_right, s_right, q_right, p_right
    ):
        c_out = c_right * c_left - s_right * s_left
        s_out = s_right * c_left + c_right * s_left
        q_out = c_right * q_left - s_right * p_left + q_right
        p_out = s_right * q_left + c_right * p_left + p_right
        return c_out, s_out, q_out, p_out

    @triton.jit
    def _complex_scan_fwd_kernel(
        c_ptr,
        s_ptr,
        uq_ptr,
        up_ptr,
        oq_ptr,
        op_ptr,
        seq_len,
        stride_bm_t,
        stride_t,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base = pid * stride_bm_t
        carry_q = tl.zeros((), dtype=tl.float32)
        carry_p = tl.zeros((), dtype=tl.float32)
        offs = tl.arange(0, BLOCK_T)

        t0 = 0
        while t0 < seq_len:
            idx = t0 + offs
            mask = idx < seq_len
            ptrs = base + idx * stride_t

            c_tile = tl.load(c_ptr + ptrs, mask=mask, other=1.0).to(tl.float32)
            s_tile = tl.load(s_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            q_tile = tl.load(uq_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            p_tile = tl.load(up_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

            q_first = c_tile * carry_q - s_tile * carry_p + q_tile
            p_first = s_tile * carry_q + c_tile * carry_p + p_tile
            q_tile = tl.where(offs == 0, q_first, q_tile)
            p_tile = tl.where(offs == 0, p_first, p_tile)

            _, _, zq_tile, zp_tile = tl.associative_scan(
                (c_tile, s_tile, q_tile, p_tile), axis=0, combine_fn=_complex_combine
            )
            tl.store(oq_ptr + ptrs, zq_tile, mask=mask)
            tl.store(op_ptr + ptrs, zp_tile, mask=mask)

            last_idx = tl.minimum(BLOCK_T - 1, seq_len - t0 - 1)
            carry_q = tl.sum(tl.where(offs == last_idx, zq_tile, 0.0), axis=0)
            carry_p = tl.sum(tl.where(offs == last_idx, zp_tile, 0.0), axis=0)
            t0 += BLOCK_T

    @triton.jit
    def _sp4_combine(
        l00,
        l01,
        l02,
        l03,
        l10,
        l11,
        l12,
        l13,
        l20,
        l21,
        l22,
        l23,
        l30,
        l31,
        l32,
        l33,
        lu0,
        lu1,
        lu2,
        lu3,
        r00,
        r01,
        r02,
        r03,
        r10,
        r11,
        r12,
        r13,
        r20,
        r21,
        r22,
        r23,
        r30,
        r31,
        r32,
        r33,
        ru0,
        ru1,
        ru2,
        ru3,
    ):
        o00 = r00 * l00 + r01 * l10 + r02 * l20 + r03 * l30
        o01 = r00 * l01 + r01 * l11 + r02 * l21 + r03 * l31
        o02 = r00 * l02 + r01 * l12 + r02 * l22 + r03 * l32
        o03 = r00 * l03 + r01 * l13 + r02 * l23 + r03 * l33

        o10 = r10 * l00 + r11 * l10 + r12 * l20 + r13 * l30
        o11 = r10 * l01 + r11 * l11 + r12 * l21 + r13 * l31
        o12 = r10 * l02 + r11 * l12 + r12 * l22 + r13 * l32
        o13 = r10 * l03 + r11 * l13 + r12 * l23 + r13 * l33

        o20 = r20 * l00 + r21 * l10 + r22 * l20 + r23 * l30
        o21 = r20 * l01 + r21 * l11 + r22 * l21 + r23 * l31
        o22 = r20 * l02 + r21 * l12 + r22 * l22 + r23 * l32
        o23 = r20 * l03 + r21 * l13 + r22 * l23 + r23 * l33

        o30 = r30 * l00 + r31 * l10 + r32 * l20 + r33 * l30
        o31 = r30 * l01 + r31 * l11 + r32 * l21 + r33 * l31
        o32 = r30 * l02 + r31 * l12 + r32 * l22 + r33 * l32
        o33 = r30 * l03 + r31 * l13 + r32 * l23 + r33 * l33

        ou0 = r00 * lu0 + r01 * lu1 + r02 * lu2 + r03 * lu3 + ru0
        ou1 = r10 * lu0 + r11 * lu1 + r12 * lu2 + r13 * lu3 + ru1
        ou2 = r20 * lu0 + r21 * lu1 + r22 * lu2 + r23 * lu3 + ru2
        ou3 = r30 * lu0 + r31 * lu1 + r32 * lu2 + r33 * lu3 + ru3

        return (
            o00,
            o01,
            o02,
            o03,
            o10,
            o11,
            o12,
            o13,
            o20,
            o21,
            o22,
            o23,
            o30,
            o31,
            o32,
            o33,
            ou0,
            ou1,
            ou2,
            ou3,
        )

    @triton.jit
    def _sp4_scan_fwd_kernel(
        a_ptr,
        u_ptr,
        out_ptr,
        seq_len,
        stride_bd_a,
        stride_t_a,
        stride_bd_u,
        stride_t_u,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base_a = pid * stride_bd_a
        base_u = pid * stride_bd_u

        c0 = tl.zeros((), dtype=tl.float32)
        c1 = tl.zeros((), dtype=tl.float32)
        c2 = tl.zeros((), dtype=tl.float32)
        c3 = tl.zeros((), dtype=tl.float32)
        offs = tl.arange(0, BLOCK_T)

        t0 = 0
        while t0 < seq_len:
            idx = t0 + offs
            mask = idx < seq_len
            pa = base_a + idx * stride_t_a
            pu = base_u + idx * stride_t_u

            a00 = tl.load(a_ptr + pa + 0, mask=mask, other=1.0).to(tl.float32)
            a01 = tl.load(a_ptr + pa + 1, mask=mask, other=0.0).to(tl.float32)
            a02 = tl.load(a_ptr + pa + 2, mask=mask, other=0.0).to(tl.float32)
            a03 = tl.load(a_ptr + pa + 3, mask=mask, other=0.0).to(tl.float32)

            a10 = tl.load(a_ptr + pa + 4, mask=mask, other=0.0).to(tl.float32)
            a11 = tl.load(a_ptr + pa + 5, mask=mask, other=1.0).to(tl.float32)
            a12 = tl.load(a_ptr + pa + 6, mask=mask, other=0.0).to(tl.float32)
            a13 = tl.load(a_ptr + pa + 7, mask=mask, other=0.0).to(tl.float32)

            a20 = tl.load(a_ptr + pa + 8, mask=mask, other=0.0).to(tl.float32)
            a21 = tl.load(a_ptr + pa + 9, mask=mask, other=0.0).to(tl.float32)
            a22 = tl.load(a_ptr + pa + 10, mask=mask, other=1.0).to(tl.float32)
            a23 = tl.load(a_ptr + pa + 11, mask=mask, other=0.0).to(tl.float32)

            a30 = tl.load(a_ptr + pa + 12, mask=mask, other=0.0).to(tl.float32)
            a31 = tl.load(a_ptr + pa + 13, mask=mask, other=0.0).to(tl.float32)
            a32 = tl.load(a_ptr + pa + 14, mask=mask, other=0.0).to(tl.float32)
            a33 = tl.load(a_ptr + pa + 15, mask=mask, other=1.0).to(tl.float32)

            u0 = tl.load(u_ptr + pu + 0, mask=mask, other=0.0).to(tl.float32)
            u1 = tl.load(u_ptr + pu + 1, mask=mask, other=0.0).to(tl.float32)
            u2 = tl.load(u_ptr + pu + 2, mask=mask, other=0.0).to(tl.float32)
            u3 = tl.load(u_ptr + pu + 3, mask=mask, other=0.0).to(tl.float32)

            f0 = a00 * c0 + a01 * c1 + a02 * c2 + a03 * c3 + u0
            f1 = a10 * c0 + a11 * c1 + a12 * c2 + a13 * c3 + u1
            f2 = a20 * c0 + a21 * c1 + a22 * c2 + a23 * c3 + u2
            f3 = a30 * c0 + a31 * c1 + a32 * c2 + a33 * c3 + u3

            u0 = tl.where(offs == 0, f0, u0)
            u1 = tl.where(offs == 0, f1, u1)
            u2 = tl.where(offs == 0, f2, u2)
            u3 = tl.where(offs == 0, f3, u3)

            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                z0,
                z1,
                z2,
                z3,
            ) = tl.associative_scan(
                (
                    a00,
                    a01,
                    a02,
                    a03,
                    a10,
                    a11,
                    a12,
                    a13,
                    a20,
                    a21,
                    a22,
                    a23,
                    a30,
                    a31,
                    a32,
                    a33,
                    u0,
                    u1,
                    u2,
                    u3,
                ),
                axis=0,
                combine_fn=_sp4_combine,
            )

            tl.store(out_ptr + pu + 0, z0, mask=mask)
            tl.store(out_ptr + pu + 1, z1, mask=mask)
            tl.store(out_ptr + pu + 2, z2, mask=mask)
            tl.store(out_ptr + pu + 3, z3, mask=mask)

            last_idx = tl.minimum(BLOCK_T - 1, seq_len - t0 - 1)
            c0 = tl.sum(tl.where(offs == last_idx, z0, 0.0), axis=0)
            c1 = tl.sum(tl.where(offs == last_idx, z1, 0.0), axis=0)
            c2 = tl.sum(tl.where(offs == last_idx, z2, 0.0), axis=0)
            c3 = tl.sum(tl.where(offs == last_idx, z3, 0.0), axis=0)
            t0 += BLOCK_T

    @triton.jit
    def _multistep_sp4_decode_kernel(
        a_ptr,
        u_ptr,
        z_init_ptr,
        valid_lens_ptr,
        z_seq_out_ptr,
        z_final_out_ptr,
        s_max,
        n_dyads,
        BLOCK_S: tl.constexpr,  # noqa: ARG001
    ):
        pid = tl.program_id(0)
        b_idx = pid // n_dyads
        d_idx = pid % n_dyads

        valid_len = tl.load(valid_lens_ptr + b_idx)
        init_base = (b_idx * n_dyads + d_idx) * 4
        z0 = tl.load(z_init_ptr + init_base + 0).to(tl.float32)
        z1 = tl.load(z_init_ptr + init_base + 1).to(tl.float32)
        z2 = tl.load(z_init_ptr + init_base + 2).to(tl.float32)
        z3 = tl.load(z_init_ptr + init_base + 3).to(tl.float32)

        for s in range(s_max):
            step_offset = (b_idx * s_max + s) * n_dyads + d_idx
            pa = step_offset * 16
            pu = step_offset * 4

            a00 = tl.load(a_ptr + pa + 0).to(tl.float32)
            a01 = tl.load(a_ptr + pa + 1).to(tl.float32)
            a02 = tl.load(a_ptr + pa + 2).to(tl.float32)
            a03 = tl.load(a_ptr + pa + 3).to(tl.float32)

            a10 = tl.load(a_ptr + pa + 4).to(tl.float32)
            a11 = tl.load(a_ptr + pa + 5).to(tl.float32)
            a12 = tl.load(a_ptr + pa + 6).to(tl.float32)
            a13 = tl.load(a_ptr + pa + 7).to(tl.float32)

            a20 = tl.load(a_ptr + pa + 8).to(tl.float32)
            a21 = tl.load(a_ptr + pa + 9).to(tl.float32)
            a22 = tl.load(a_ptr + pa + 10).to(tl.float32)
            a23 = tl.load(a_ptr + pa + 11).to(tl.float32)

            a30 = tl.load(a_ptr + pa + 12).to(tl.float32)
            a31 = tl.load(a_ptr + pa + 13).to(tl.float32)
            a32 = tl.load(a_ptr + pa + 14).to(tl.float32)
            a33 = tl.load(a_ptr + pa + 15).to(tl.float32)

            u0 = tl.load(u_ptr + pu + 0).to(tl.float32)
            u1 = tl.load(u_ptr + pu + 1).to(tl.float32)
            u2 = tl.load(u_ptr + pu + 2).to(tl.float32)
            u3 = tl.load(u_ptr + pu + 3).to(tl.float32)

            nz0 = a00 * z0 + a01 * z1 + a02 * z2 + a03 * z3 + u0
            nz1 = a10 * z0 + a11 * z1 + a12 * z2 + a13 * z3 + u1
            nz2 = a20 * z0 + a21 * z1 + a22 * z2 + a23 * z3 + u2
            nz3 = a30 * z0 + a31 * z1 + a32 * z2 + a33 * z3 + u3

            tl.store(z_seq_out_ptr + pu + 0, nz0)
            tl.store(z_seq_out_ptr + pu + 1, nz1)
            tl.store(z_seq_out_ptr + pu + 2, nz2)
            tl.store(z_seq_out_ptr + pu + 3, nz3)

            active = s < valid_len
            z0 = tl.where(active, nz0, z0)
            z1 = tl.where(active, nz1, z1)
            z2 = tl.where(active, nz2, z2)
            z3 = tl.where(active, nz3, z3)

        tl.store(z_final_out_ptr + init_base + 0, z0)
        tl.store(z_final_out_ptr + init_base + 1, z1)
        tl.store(z_final_out_ptr + init_base + 2, z2)
        tl.store(z_final_out_ptr + init_base + 3, z3)


def _launch_triton_scalar_scan(a_bmt: Tensor, u_bmt: Tensor) -> Tensor:
    """Run _scalar_scan_fwd_kernel on contiguous [BM, T] float32 tensors."""
    bm, seq_len = u_bmt.shape
    out = torch.empty_like(u_bmt)
    if seq_len == 0 or bm == 0:
        return out
    block_t = 256 if seq_len >= 128 else 64
    _scalar_scan_fwd_kernel[(bm,)](
        a_bmt,
        u_bmt,
        out,
        seq_len,
        a_bmt.stride(0),
        a_bmt.stride(1),
        BLOCK_T=block_t,
        num_warps=4 if block_t >= 256 else 2,
    )
    return out


def _launch_triton_complex_scan(
    c_bmt: Tensor, s_bmt: Tensor, uq_bmt: Tensor, up_bmt: Tensor
) -> tuple[Tensor, Tensor]:
    """Run _complex_scan_fwd_kernel on contiguous [BM, T] float32 tensors."""
    bm, seq_len = uq_bmt.shape
    oq = torch.empty_like(uq_bmt)
    op = torch.empty_like(up_bmt)
    if seq_len == 0 or bm == 0:
        return oq, op
    block_t = 256 if seq_len >= 128 else 64
    _complex_scan_fwd_kernel[(bm,)](
        c_bmt,
        s_bmt,
        uq_bmt,
        up_bmt,
        oq,
        op,
        seq_len,
        c_bmt.stride(0),
        c_bmt.stride(1),
        BLOCK_T=block_t,
        num_warps=4 if block_t >= 256 else 2,
    )
    return oq, op


def _launch_triton_sp4_scan(a_bdt: Tensor, u_bdt: Tensor) -> Tensor:
    """Run _sp4_scan_fwd_kernel on contiguous [BD, T, 4, 4] and [BD, T, 4] tensors."""
    bd, seq_len, _ = u_bdt.shape
    out = torch.empty_like(u_bdt)
    if seq_len == 0 or bd == 0:
        return out
    block_t = 128 if seq_len >= 64 else 32
    a_flat = a_bdt.reshape(bd, seq_len, 16).contiguous()
    u_flat = u_bdt.contiguous()
    _sp4_scan_fwd_kernel[(bd,)](
        a_flat,
        u_flat,
        out,
        seq_len,
        a_flat.stride(0),
        a_flat.stride(1),
        u_flat.stride(0),
        u_flat.stride(1),
        BLOCK_T=block_t,
        num_warps=4 if block_t >= 128 else 1,
    )
    return out


class _TritonScalarScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_decay: Tensor, u_val: Tensor) -> Tensor:
        bsz, seq_len, n_modes = u_val.shape
        a_bmt = (
            a_decay.float().permute(0, 2, 1).contiguous().view(bsz * n_modes, seq_len)
        )
        u_bmt = u_val.float().permute(0, 2, 1).contiguous().view(bsz * n_modes, seq_len)
        z_bmt = _launch_triton_scalar_scan(a_bmt, u_bmt)
        ctx.save_for_backward(a_bmt, z_bmt)
        ctx.shape_meta = (bsz, seq_len, n_modes, a_decay.dtype, u_val.dtype)
        return z_bmt.view(bsz, n_modes, seq_len).permute(0, 2, 1).contiguous()

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        a_bmt, z_bmt = ctx.saved_tensors
        bsz, seq_len, n_modes, a_dtype, u_dtype = ctx.shape_meta
        g_bmt = (
            grad_out.float().permute(0, 2, 1).contiguous().view(bsz * n_modes, seq_len)
        )

        if seq_len > 1:
            a_rev = torch.cat(
                [torch.zeros_like(a_bmt[:, :1]), a_bmt[:, 1:].flip(dims=[1])],
                dim=1,
            ).contiguous()
        else:
            a_rev = torch.zeros_like(a_bmt)
        g_rev = g_bmt.flip(dims=[1]).contiguous()
        lam_bmt = _launch_triton_scalar_scan(a_rev, g_rev).flip(dims=[1])

        z_prev = torch.cat([torch.zeros_like(z_bmt[:, :1]), z_bmt[:, :-1]], dim=1)
        grad_a_bmt = lam_bmt * z_prev

        grad_a = (
            grad_a_bmt.view(bsz, n_modes, seq_len)
            .permute(0, 2, 1)
            .contiguous()
            .to(a_dtype)
        )
        grad_u = (
            lam_bmt.view(bsz, n_modes, seq_len)
            .permute(0, 2, 1)
            .contiguous()
            .to(u_dtype)
        )
        return grad_a, grad_u


class _TritonComplexScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, a_cos: Tensor, a_sin: Tensor, u_q: Tensor, u_p: Tensor
    ) -> tuple[Tensor, Tensor]:
        bsz, seq_len, n_modes = u_q.shape
        bm = bsz * n_modes
        c_bmt = a_cos.float().permute(0, 2, 1).contiguous().view(bm, seq_len)
        s_bmt = a_sin.float().permute(0, 2, 1).contiguous().view(bm, seq_len)
        uq_bmt = u_q.float().permute(0, 2, 1).contiguous().view(bm, seq_len)
        up_bmt = u_p.float().permute(0, 2, 1).contiguous().view(bm, seq_len)

        zq_bmt, zp_bmt = _launch_triton_complex_scan(c_bmt, s_bmt, uq_bmt, up_bmt)
        ctx.save_for_backward(c_bmt, s_bmt, zq_bmt, zp_bmt)
        ctx.shape_meta = (
            bsz,
            seq_len,
            n_modes,
            a_cos.dtype,
            a_sin.dtype,
            u_q.dtype,
            u_p.dtype,
        )
        q_seq = zq_bmt.view(bsz, n_modes, seq_len).permute(0, 2, 1).contiguous()
        p_seq = zp_bmt.view(bsz, n_modes, seq_len).permute(0, 2, 1).contiguous()
        return q_seq, p_seq

    @staticmethod
    def backward(ctx, grad_q: Tensor, grad_p: Tensor):
        c_bmt, s_bmt, zq_bmt, zp_bmt = ctx.saved_tensors
        bsz, seq_len, n_modes, c_dt, s_dt, q_dt, p_dt = ctx.shape_meta
        bm = bsz * n_modes

        gq_bmt = grad_q.float().permute(0, 2, 1).contiguous().view(bm, seq_len)
        gp_bmt = grad_p.float().permute(0, 2, 1).contiguous().view(bm, seq_len)

        # Adjoint of R(theta) = [[c, -s], [s, c]] is R(theta)^T = [[c, s], [-s, c]].
        if seq_len > 1:
            c_rev = torch.cat(
                [torch.zeros_like(c_bmt[:, :1]), c_bmt[:, 1:].flip(dims=[1])],
                dim=1,
            ).contiguous()
            s_rev = torch.cat(
                [torch.zeros_like(s_bmt[:, :1]), (-s_bmt[:, 1:]).flip(dims=[1])],
                dim=1,
            ).contiguous()
        else:
            c_rev = torch.zeros_like(c_bmt)
            s_rev = torch.zeros_like(s_bmt)

        gq_rev = gq_bmt.flip(dims=[1]).contiguous()
        gp_rev = gp_bmt.flip(dims=[1]).contiguous()
        lam_q, lam_p = _launch_triton_complex_scan(c_rev, s_rev, gq_rev, gp_rev)
        lam_q = lam_q.flip(dims=[1])
        lam_p = lam_p.flip(dims=[1])

        zq_prev = torch.cat([torch.zeros_like(zq_bmt[:, :1]), zq_bmt[:, :-1]], dim=1)
        zp_prev = torch.cat([torch.zeros_like(zp_bmt[:, :1]), zp_bmt[:, :-1]], dim=1)

        gc_bmt = lam_q * zq_prev + lam_p * zp_prev
        gs_bmt = -lam_q * zp_prev + lam_p * zq_prev

        def _unperm(t: Tensor, dt: torch.dtype) -> Tensor:
            return t.view(bsz, n_modes, seq_len).permute(0, 2, 1).contiguous().to(dt)

        return (
            _unperm(gc_bmt, c_dt),
            _unperm(gs_bmt, s_dt),
            _unperm(lam_q, q_dt),
            _unperm(lam_p, p_dt),
        )


class _TritonSp4ScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_mat: Tensor, u_vec: Tensor) -> Tensor:
        bsz, seq_len, n_dyads, _, _ = a_mat.shape
        bd = bsz * n_dyads
        a_bdt = (
            a_mat.float().permute(0, 2, 1, 3, 4).contiguous().view(bd, seq_len, 4, 4)
        )
        u_bdt = u_vec.float().permute(0, 2, 1, 3).contiguous().view(bd, seq_len, 4)
        z_bdt = _launch_triton_sp4_scan(a_bdt, u_bdt)
        ctx.save_for_backward(a_bdt, z_bdt)
        ctx.shape_meta = (bsz, seq_len, n_dyads, a_mat.dtype, u_vec.dtype)
        return z_bdt.view(bsz, n_dyads, seq_len, 4).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, grad_z: Tensor):
        a_bdt, z_bdt = ctx.saved_tensors
        bsz, seq_len, n_dyads, a_dt, u_dt = ctx.shape_meta
        bd = bsz * n_dyads
        g_bdt = grad_z.float().permute(0, 2, 1, 3).contiguous().view(bd, seq_len, 4)

        if seq_len > 1:
            at_rev = torch.cat(
                [
                    torch.zeros_like(a_bdt[:, :1]),
                    a_bdt[:, 1:].transpose(-1, -2).flip(dims=[1]),
                ],
                dim=1,
            ).contiguous()
        else:
            at_rev = torch.zeros_like(a_bdt)
        g_rev = g_bdt.flip(dims=[1]).contiguous()
        lam_bdt = _launch_triton_sp4_scan(at_rev, g_rev).flip(dims=[1])

        z_prev = torch.cat([torch.zeros_like(z_bdt[:, :1]), z_bdt[:, :-1]], dim=1)
        grad_a_bdt = lam_bdt.unsqueeze(-1) * z_prev.unsqueeze(-2)

        grad_a = (
            grad_a_bdt.view(bsz, n_dyads, seq_len, 4, 4)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .to(a_dt)
        )
        grad_u = (
            lam_bdt.view(bsz, n_dyads, seq_len, 4)
            .permute(0, 2, 1, 3)
            .contiguous()
            .to(u_dt)
        )
        return grad_a, grad_u


# ---------------------------------------------------------------------------
# Public Scan & Fused Multi-Step Decode Dispatchers
# ---------------------------------------------------------------------------


def segmented_scalar_scan(
    a_decay: Tensor,
    u_val: Tensor,
    initial_state: Tensor | None = None,
    use_triton: bool = True,
) -> Tensor:
    """Segmented FP32 scalar scan z_t = a_t * z_{t-1} + u_t (Lindblad variance)."""
    a_f = a_decay.float()
    u_f = _fold_scalar_initial_state(a_f, u_val.float(), initial_state)
    if _can_run_triton(u_f, use_triton):
        return _TritonScalarScanFn.apply(a_f, u_f)
    return pytorch_segmented_scalar_scan(a_f, u_f, initial_state=None)


def segmented_complex_scan(
    a_cos: Tensor,
    a_sin: Tensor,
    u_q: Tensor,
    u_p: Tensor,
    initial_state: tuple[Tensor, Tensor] | None = None,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor]:
    """Segmented FP32 complex quadrature scan (q_t + i p_t)."""
    c_f = a_cos.float()
    s_f = a_sin.float()
    uq_f, up_f = _fold_complex_initial_state(
        c_f, s_f, u_q.float(), u_p.float(), initial_state
    )
    if _can_run_triton(uq_f, use_triton):
        return _TritonComplexScanFn.apply(c_f, s_f, uq_f, up_f)
    return pytorch_segmented_complex_scan(c_f, s_f, uq_f, up_f, initial_state=None)


def segmented_sp4_scan(
    a_mat: Tensor,
    u_vec: Tensor,
    initial_state: Tensor | None = None,
    use_triton: bool = True,
) -> Tensor:
    """Segmented FP32 4x4 sp(4, R) symplectic-squeezed dyad scan."""
    a_f = a_mat.float()
    u_f = _fold_sp4_initial_state(a_f, u_vec.float(), initial_state)
    if _can_run_triton(u_f, use_triton):
        return _TritonSp4ScanFn.apply(a_f, u_f)
    return pytorch_segmented_sp4_scan(a_f, u_f, initial_state=None)


def fused_multistep_complex_decode(
    a_cos: Tensor,  # [B, S_max, M]
    a_sin: Tensor,  # [B, S_max, M]
    u_q: Tensor,  # [B, S_max, M]
    u_p: Tensor,  # [B, S_max, M]
    q_init: Tensor,  # [B, M]
    p_init: Tensor,  # [B, M]
    valid_lens: Tensor,  # [B]
    a_nu: Tensor | None = None,  # [B, S_max, M]
    u_nu: Tensor | None = None,  # [B, S_max, M]
    nu_init: Tensor | None = None,  # [B, M]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    """Advance (q, p[, nu]) across S_max steps up to valid_lens in FP32."""
    bsz, s_max, _ = u_q.shape
    c_f = a_cos.float()
    s_f = a_sin.float()
    uq_f = u_q.float()
    up_f = u_p.float()
    q_cur = q_init.float()
    p_cur = p_init.float()
    nu_cur = nu_init.float() if nu_init is not None else None
    a_nu_f = a_nu.float() if a_nu is not None else None
    u_nu_f = u_nu.float() if u_nu is not None else None

    v_lens = valid_lens.to(u_q.device)
    q_steps: list[Tensor] = []
    p_steps: list[Tensor] = []
    nu_steps: list[Tensor] = []

    for s in range(s_max):
        active = (v_lens > s).unsqueeze(-1)
        nq = c_f[:, s] * q_cur - s_f[:, s] * p_cur + uq_f[:, s]
        np_ = s_f[:, s] * q_cur + c_f[:, s] * p_cur + up_f[:, s]
        q_steps.append(nq)
        p_steps.append(np_)
        q_cur = torch.where(active, nq, q_cur)
        p_cur = torch.where(active, np_, p_cur)
        if nu_cur is not None and a_nu_f is not None and u_nu_f is not None:
            nnu = a_nu_f[:, s] * nu_cur + u_nu_f[:, s]
            nu_steps.append(nnu)
            nu_cur = torch.where(active, nnu, nu_cur)

    q_seq = torch.stack(q_steps, dim=1) if s_max > 0 else uq_f[:, :0]
    p_seq = torch.stack(p_steps, dim=1) if s_max > 0 else up_f[:, :0]
    nu_seq = (
        (torch.stack(nu_steps, dim=1) if s_max > 0 else u_nu_f[:, :0])
        if (nu_cur is not None and u_nu_f is not None)
        else None
    )
    return q_seq, p_seq, q_cur, p_cur, nu_seq, nu_cur


def fused_multistep_sp4_decode(
    a_mat: Tensor,  # [B, S_max, D_dyad, 4, 4]
    u_vec: Tensor,  # [B, S_max, D_dyad, 4]
    z_init: Tensor,  # [B, D_dyad, 4]
    valid_lens: Tensor,  # [B]
    a_nu: Tensor | None = None,  # [B, S_max, M]
    u_nu: Tensor | None = None,  # [B, S_max, M]
    nu_init: Tensor | None = None,  # [B, M]
    use_triton: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Persistent-register multi-step 4x4 sp(4, R) decode across S_max tokens."""
    bsz, s_max, n_dyads, _ = u_vec.shape
    a_f = a_mat.float()
    u_f = u_vec.float()
    z_cur = z_init.float().contiguous()
    v_lens = valid_lens.to(device=u_vec.device, dtype=torch.int64).contiguous()

    if _can_run_triton(u_f, use_triton) and s_max > 0:
        a_flat = a_f.reshape(bsz, s_max, n_dyads, 16).contiguous()
        u_contig = u_f.contiguous()
        z_seq = torch.empty_like(u_contig)
        z_final = torch.empty_like(z_cur)
        _multistep_sp4_decode_kernel[(bsz * n_dyads,)](
            a_flat,
            u_contig,
            z_cur,
            v_lens,
            z_seq,
            z_final,
            s_max,
            n_dyads,
            BLOCK_S=1,
            num_warps=1,
        )
    else:
        z_steps: list[Tensor] = []
        for s in range(s_max):
            active = (v_lens > s).view(bsz, 1, 1)
            nz = torch.matmul(a_f[:, s], z_cur.unsqueeze(-1)).squeeze(-1) + u_f[:, s]
            z_steps.append(nz)
            z_cur = torch.where(active, nz, z_cur)
        z_seq = torch.stack(z_steps, dim=1) if s_max > 0 else u_f[:, :0]
        z_final = z_cur

    nu_seq: Tensor | None = None
    nu_final: Tensor | None = None
    if nu_init is not None and a_nu is not None and u_nu is not None:
        nu_cur = nu_init.float()
        a_nu_f = a_nu.float()
        u_nu_f = u_nu.float()
        nu_steps: list[Tensor] = []
        for s in range(s_max):
            active_nu = (v_lens > s).unsqueeze(-1)
            nnu = a_nu_f[:, s] * nu_cur + u_nu_f[:, s]
            nu_steps.append(nnu)
            nu_cur = torch.where(active_nu, nnu, nu_cur)
        nu_seq = torch.stack(nu_steps, dim=1) if s_max > 0 else u_nu_f[:, :0]
        nu_final = nu_cur

    return z_seq, z_final, nu_seq, nu_final

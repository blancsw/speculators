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
    "fused_2d_osc_dynamics_multistep_decode",
    "fused_dyadic_rff_2stage_trig",
    "fused_engram_hash_rmsnorm",
    "fused_interferometer_born_obs",
    "fused_multistep_complex_decode",
    "fused_multistep_sp4_decode",
    "fused_rmsnorm_trig_mod",
    "pytorch_segmented_2d_osc_scan",
    "pytorch_segmented_complex_scan",
    "pytorch_segmented_scalar_scan",
    "pytorch_segmented_sp4_scan",
    "segmented_2d_osc_scan",
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


def pytorch_segmented_2d_osc_scan(
    a_cos: Tensor,
    a_sin: Tensor,
    u_q: Tensor,
    u_p: Tensor,
    a_nu: Tensor | None = None,
    u_nu: Tensor | None = None,
    reset_mask: Tensor | None = None,
    initial_state: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Work-parallel O(T log T) FP32 reference scan for 2D oscillator + Lindblad nu."""
    c_f = a_cos.float()
    s_f = a_sin.float()
    uq_f = u_q.float()
    up_f = u_p.float()
    anu_f = a_nu.float() if a_nu is not None else None
    unu_f = u_nu.float() if u_nu is not None else None

    if reset_mask is not None:
        keep = (~reset_mask).to(torch.float32).unsqueeze(-1)
        c_f = c_f * keep
        s_f = s_f * keep
        if anu_f is not None:
            anu_f = anu_f * keep

    c_init: tuple[Tensor, Tensor] | None = None
    nu_init: Tensor | None = None
    if initial_state is not None and len(initial_state) >= 2:
        if initial_state[0] is not None and initial_state[1] is not None:
            c_init = (initial_state[0], initial_state[1])
        if len(initial_state) >= 3 and initial_state[2] is not None:
            nu_init = initial_state[2]

    q_seq, p_seq = pytorch_segmented_complex_scan(
        c_f, s_f, uq_f, up_f, initial_state=c_init
    )
    nu_seq: Tensor | None = None
    if anu_f is not None and unu_f is not None:
        nu_seq = pytorch_segmented_scalar_scan(anu_f, unu_f, initial_state=nu_init)
    return q_seq, p_seq, nu_seq


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

    @triton.jit
    def _2d_osc_nu_combine(
        c_left,
        s_left,
        q_left,
        p_left,
        anu_left,
        unu_left,
        c_right,
        s_right,
        q_right,
        p_right,
        anu_right,
        unu_right,
    ):
        c_out = c_right * c_left - s_right * s_left
        s_out = s_right * c_left + c_right * s_left
        q_out = c_right * q_left - s_right * p_left + q_right
        p_out = s_right * q_left + c_right * p_left + p_right
        anu_out = anu_right * anu_left
        unu_out = anu_right * unu_left + unu_right
        return c_out, s_out, q_out, p_out, anu_out, unu_out

    @triton.jit
    def _2d_osc_scan_fwd_kernel(
        c_ptr,
        s_ptr,
        uq_ptr,
        up_ptr,
        anu_ptr,
        unu_ptr,
        reset_ptr,
        q0_ptr,
        p0_ptr,
        nu0_ptr,
        oq_ptr,
        op_ptr,
        onu_ptr,
        seq_len,
        n_modes,
        HAS_NU: tl.constexpr,
        HAS_RESET: tl.constexpr,
        HAS_INIT_QP: tl.constexpr,
        HAS_INIT_NU: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        m_tile = tl.program_id(1)
        offs_t = tl.arange(0, BLOCK_T)
        offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_modes

        if HAS_INIT_QP:
            init_ptrs = b_idx * n_modes + offs_m
            carry_q = tl.load(q0_ptr + init_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            carry_p = tl.load(p0_ptr + init_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        else:
            carry_q = tl.zeros((BLOCK_M,), dtype=tl.float32)
            carry_p = tl.zeros((BLOCK_M,), dtype=tl.float32)

        if HAS_NU:
            if HAS_INIT_NU:
                init_nu_ptrs = b_idx * n_modes + offs_m
                carry_nu = tl.load(nu0_ptr + init_nu_ptrs, mask=mask_m, other=0.0).to(
                    tl.float32
                )
            else:
                carry_nu = tl.zeros((BLOCK_M,), dtype=tl.float32)

        is_first = offs_t[:, None] == 0
        t0 = 0
        while t0 < seq_len:
            idx_t = t0 + offs_t
            mask_t = idx_t < seq_len
            mask_tm = mask_t[:, None] & mask_m[None, :]
            ptrs = (b_idx * seq_len + idx_t[:, None]) * n_modes + offs_m[None, :]

            c_tile = tl.load(c_ptr + ptrs, mask=mask_tm, other=1.0).to(tl.float32)
            s_tile = tl.load(s_ptr + ptrs, mask=mask_tm, other=0.0).to(tl.float32)
            q_tile = tl.load(uq_ptr + ptrs, mask=mask_tm, other=0.0).to(tl.float32)
            p_tile = tl.load(up_ptr + ptrs, mask=mask_tm, other=0.0).to(tl.float32)

            if HAS_RESET:
                rst = tl.load(
                    reset_ptr + b_idx * seq_len + idx_t, mask=mask_t, other=0
                ).to(tl.int1)
                c_tile = tl.where(rst[:, None], 0.0, c_tile)
                s_tile = tl.where(rst[:, None], 0.0, s_tile)

            q_first = c_tile * carry_q[None, :] - s_tile * carry_p[None, :] + q_tile
            p_first = s_tile * carry_q[None, :] + c_tile * carry_p[None, :] + p_tile
            q_tile = tl.where(is_first, q_first, q_tile)
            p_tile = tl.where(is_first, p_first, p_tile)

            last_idx = tl.minimum(BLOCK_T - 1, seq_len - t0 - 1)
            is_last = offs_t[:, None] == last_idx

            if HAS_NU:
                anu_tile = tl.load(anu_ptr + ptrs, mask=mask_tm, other=1.0).to(
                    tl.float32
                )
                unu_tile = tl.load(unu_ptr + ptrs, mask=mask_tm, other=0.0).to(
                    tl.float32
                )
                if HAS_RESET:
                    anu_tile = tl.where(rst[:, None], 0.0, anu_tile)
                nu_first = anu_tile * carry_nu[None, :] + unu_tile
                unu_tile = tl.where(is_first, nu_first, unu_tile)
                _, _, zq_tile, zp_tile, _, znu_tile = tl.associative_scan(
                    (c_tile, s_tile, q_tile, p_tile, anu_tile, unu_tile),
                    axis=0,
                    combine_fn=_2d_osc_nu_combine,
                )
                tl.store(oq_ptr + ptrs, zq_tile, mask=mask_tm)
                tl.store(op_ptr + ptrs, zp_tile, mask=mask_tm)
                tl.store(onu_ptr + ptrs, znu_tile, mask=mask_tm)
                carry_nu = tl.sum(tl.where(is_last, znu_tile, 0.0), axis=0)
            else:
                _, _, zq_tile, zp_tile = tl.associative_scan(
                    (c_tile, s_tile, q_tile, p_tile),
                    axis=0,
                    combine_fn=_complex_combine,
                )
                tl.store(oq_ptr + ptrs, zq_tile, mask=mask_tm)
                tl.store(op_ptr + ptrs, zp_tile, mask=mask_tm)

            carry_q = tl.sum(tl.where(is_last, zq_tile, 0.0), axis=0)
            carry_p = tl.sum(tl.where(is_last, zp_tile, 0.0), axis=0)
            t0 += BLOCK_T

    @triton.jit
    def _multistep_2d_osc_decode_kernel(
        c_ptr,
        s_ptr,
        uq_ptr,
        up_ptr,
        q_init_ptr,
        p_init_ptr,
        valid_lens_ptr,
        anu_ptr,
        unu_ptr,
        nu_init_ptr,
        q_seq_ptr,
        p_seq_ptr,
        q_fin_ptr,
        p_fin_ptr,
        nu_seq_ptr,
        nu_fin_ptr,
        s_max,
        n_modes,
        HAS_NU: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        m_tile = tl.program_id(1)
        offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_modes

        valid_len = tl.load(valid_lens_ptr + b_idx)
        init_ptrs = b_idx * n_modes + offs_m
        q_cur = tl.load(q_init_ptr + init_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        p_cur = tl.load(p_init_ptr + init_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        if HAS_NU:
            nu_cur = tl.load(nu_init_ptr + init_ptrs, mask=mask_m, other=0.0).to(
                tl.float32
            )

        for s in range(s_max):
            step_ptrs = (b_idx * s_max + s) * n_modes + offs_m
            c_val = tl.load(c_ptr + step_ptrs, mask=mask_m, other=1.0).to(tl.float32)
            s_val = tl.load(s_ptr + step_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            uq_val = tl.load(uq_ptr + step_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            up_val = tl.load(up_ptr + step_ptrs, mask=mask_m, other=0.0).to(tl.float32)

            nq = c_val * q_cur - s_val * p_cur + uq_val
            np_ = s_val * q_cur + c_val * p_cur + up_val
            tl.store(q_seq_ptr + step_ptrs, nq, mask=mask_m)
            tl.store(p_seq_ptr + step_ptrs, np_, mask=mask_m)

            active = s < valid_len
            q_cur = tl.where(active, nq, q_cur)
            p_cur = tl.where(active, np_, p_cur)

            if HAS_NU:
                anu_val = tl.load(anu_ptr + step_ptrs, mask=mask_m, other=1.0).to(
                    tl.float32
                )
                unu_val = tl.load(unu_ptr + step_ptrs, mask=mask_m, other=0.0).to(
                    tl.float32
                )
                nnu = anu_val * nu_cur + unu_val
                tl.store(nu_seq_ptr + step_ptrs, nnu, mask=mask_m)
                nu_cur = tl.where(active, nnu, nu_cur)

        tl.store(q_fin_ptr + init_ptrs, q_cur, mask=mask_m)
        tl.store(p_fin_ptr + init_ptrs, p_cur, mask=mask_m)
        if HAS_NU:
            tl.store(nu_fin_ptr + init_ptrs, nu_cur, mask=mask_m)

    @triton.jit
    def _multistep_scalar_decode_kernel(
        anu_ptr,
        unu_ptr,
        nu_init_ptr,
        valid_lens_ptr,
        nu_seq_ptr,
        nu_fin_ptr,
        s_max,
        n_modes,
        BLOCK_M: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        m_tile = tl.program_id(1)
        offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_modes

        valid_len = tl.load(valid_lens_ptr + b_idx)
        init_ptrs = b_idx * n_modes + offs_m
        nu_cur = tl.load(nu_init_ptr + init_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        for s in range(s_max):
            step_ptrs = (b_idx * s_max + s) * n_modes + offs_m
            anu_val = tl.load(anu_ptr + step_ptrs, mask=mask_m, other=1.0).to(
                tl.float32
            )
            unu_val = tl.load(unu_ptr + step_ptrs, mask=mask_m, other=0.0).to(
                tl.float32
            )
            nnu = anu_val * nu_cur + unu_val
            tl.store(nu_seq_ptr + step_ptrs, nnu, mask=mask_m)
            active = s < valid_len
            nu_cur = tl.where(active, nnu, nu_cur)

        tl.store(nu_fin_ptr + init_ptrs, nu_cur, mask=mask_m)

    @triton.jit
    def _fused_2d_osc_dynamics_decode_kernel(
        rq_ptr,
        rp_ptr,
        raw_dt_ptr,
        raw_env_ptr,
        log_eta_ptr,
        omega_ptr,
        gain_q_ptr,
        gain_p_ptr,
        lind_gain_ptr,
        q_init_ptr,
        p_init_ptr,
        nu_init_ptr,
        valid_lens_ptr,
        q_seq_ptr,
        p_seq_ptr,
        q_fin_ptr,
        p_fin_ptr,
        nu_fin_ptr,
        dtheta_ptr,
        s_max,
        n_modes,
        damping_min,
        damping_max,
        HAS_NU: tl.constexpr,
        ROUND_DT_BF16: tl.constexpr,
        ROUND_DT_FP16: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        m_tile = tl.program_id(1)
        offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_modes

        valid_len = tl.load(valid_lens_ptr + b_idx)
        log_eta = tl.load(log_eta_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        eta = tl.minimum(tl.maximum(tl.exp(log_eta), damping_min), damping_max)
        omega = tl.load(omega_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        gain_q = tl.load(gain_q_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        gain_p = tl.load(gain_p_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        init_ptrs = b_idx * n_modes + offs_m
        q_cur = tl.load(q_init_ptr + init_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        p_cur = tl.load(p_init_ptr + init_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        if HAS_NU:
            lind_g = tl.load(lind_gain_ptr + offs_m, mask=mask_m, other=0.0).to(
                tl.float32
            )
            g_sq = lind_g * lind_g
            nu_cur = tl.load(nu_init_ptr + init_ptrs, mask=mask_m, other=0.0).to(
                tl.float32
            )

        for s in range(s_max):
            step_ptrs = (b_idx * s_max + s) * n_modes + offs_m
            raw_dt = tl.load(raw_dt_ptr + step_ptrs, mask=mask_m, other=0.0).to(
                tl.float32
            )
            dt_f = tl.where(raw_dt > 20.0, raw_dt, tl.log(1.0 + tl.exp(raw_dt)))
            if ROUND_DT_BF16:
                dt_f = dt_f.to(tl.bfloat16).to(tl.float32)
            elif ROUND_DT_FP16:
                dt_f = dt_f.to(tl.float16).to(tl.float32)
            dtheta_f = dt_f * omega
            tl.store(dtheta_ptr + step_ptrs, dtheta_f, mask=mask_m)

            decay = tl.exp(-eta * dt_f)
            c_val = decay * tl.cos(dtheta_f)
            s_val = decay * tl.sin(dtheta_f)

            rq_val = tl.load(rq_ptr + step_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            rp_val = tl.load(rp_ptr + step_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            uq_val = dt_f * gain_q * rq_val
            up_val = dt_f * gain_p * rp_val

            nq = c_val * q_cur - s_val * p_cur + uq_val
            np_ = s_val * q_cur + c_val * p_cur + up_val
            tl.store(q_seq_ptr + step_ptrs, nq, mask=mask_m)
            tl.store(p_seq_ptr + step_ptrs, np_, mask=mask_m)

            active = s < valid_len
            q_cur = tl.where(active, nq, q_cur)
            p_cur = tl.where(active, np_, p_cur)

            if HAS_NU:
                raw_env = tl.load(raw_env_ptr + step_ptrs, mask=mask_m, other=0.0).to(
                    tl.float32
                )
                env_sp = tl.where(
                    raw_env > 20.0, raw_env, tl.log(1.0 + tl.exp(raw_env))
                )
                anu_val = tl.exp(-2.0 * eta * dt_f)
                nnu = anu_val * nu_cur + dt_f * g_sq * env_sp
                nu_cur = tl.where(active, nnu, nu_cur)

        tl.store(q_fin_ptr + init_ptrs, q_cur, mask=mask_m)
        tl.store(p_fin_ptr + init_ptrs, p_cur, mask=mask_m)
        if HAS_NU:
            tl.store(nu_fin_ptr + init_ptrs, nu_cur, mask=mask_m)

    @triton.jit
    def _fused_engram_hash_rmsnorm_kernel(
        p1_ptr,
        p2_ptr,
        p3_ptr,
        table_ptr,
        rms_w_ptr,
        out_ptr,
        buckets,
        engram_dim,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        v1 = tl.maximum(tl.load(p1_ptr + row).to(tl.int64), 0)
        v2 = tl.maximum(tl.load(p2_ptr + row).to(tl.int64), 0)
        v3 = tl.maximum(tl.load(p3_ptr + row).to(tl.int64), 0)

        b64 = tl.full((), buckets, dtype=tl.int64)
        h0 = (v2 * 1315423911 + v1 * 2654435761 + 0x9E3779B9) % b64
        h1 = (v2 * 2246822519 + v1 * 3266489917 + 0x85EBCA6B) % b64
        h2 = (v3 * 668265263 + v2 * 374761393 + v1 * 2654435761 + 0xC2B2AE35) % b64
        h3 = (v3 * 374761393 + v2 * 1315423911 + v1 * 668265263 + 0x27D4EB2F) % b64

        offs = tl.arange(0, BLOCK_D)
        mask = offs < engram_dim
        e0 = tl.load(table_ptr + h0 * engram_dim + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        e1 = tl.load(table_ptr + h1 * engram_dim + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        e2 = tl.load(table_ptr + h2 * engram_dim + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        e3 = tl.load(table_ptr + h3 * engram_dim + offs, mask=mask, other=0.0).to(
            tl.float32
        )

        emb = 0.25 * (e0 + e1 + e2 + e3)
        msq = tl.sum(tl.where(mask, emb * emb, 0.0), axis=0) / tl.full(
            (), engram_dim, dtype=tl.float32
        )
        inv_rms = tl.rsqrt(msq + eps)
        w = tl.load(rms_w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        normed = emb * inv_rms * w
        tl.store(out_ptr + row * engram_dim + offs, normed, mask=mask)

    @triton.jit
    def _fused_interferometer_born_obs_kernel(
        q_ptr,
        p_ptr,
        theta_ptr,
        phase_ptr,
        kappa_ptr,
        obs_out_ptr,
        q_obs_out_ptr,
        p_obs_out_ptr,
        born_out_ptr,
        pvac_out_ptr,
        n_dyads,
        HAS_INTERF: tl.constexpr,
        HAS_VAC: tl.constexpr,
        WRITE_STATS: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        row = tl.program_id(0)
        n_modes = 2 * n_dyads
        offs_g = tl.arange(0, BLOCK_G)
        mask_g = offs_g < n_dyads

        base_in = row * n_modes
        q0 = tl.load(q_ptr + base_in + 2 * offs_g, mask=mask_g, other=0.0).to(
            tl.float32
        )
        q1 = tl.load(q_ptr + base_in + 2 * offs_g + 1, mask=mask_g, other=0.0).to(
            tl.float32
        )
        p0 = tl.load(p_ptr + base_in + 2 * offs_g, mask=mask_g, other=0.0).to(
            tl.float32
        )
        p1 = tl.load(p_ptr + base_in + 2 * offs_g + 1, mask=mask_g, other=0.0).to(
            tl.float32
        )

        if HAS_INTERF:
            th = tl.load(theta_ptr + offs_g, mask=mask_g, other=0.0).to(tl.float32)
            ph = tl.load(phase_ptr + offs_g, mask=mask_g, other=0.0).to(tl.float32)
            c = tl.cos(th)
            s = tl.sin(th)
            cp = tl.cos(ph)
            sp = tl.sin(ph)
            rot_q1 = cp * q1 + sp * p1
            rot_p1 = cp * p1 - sp * q1
            rot_q0 = cp * q0 - sp * p0
            rot_p0 = cp * p0 + sp * q0
            oq0 = c * q0 - s * rot_q1
            op0 = c * p0 - s * rot_p1
            oq1 = s * rot_q0 + c * q1
            op1 = s * rot_p0 + c * p1
        else:
            oq0, oq1, op0, op1 = q0, q1, p0, p1

        m_f = tl.full((), 2 * n_dyads, dtype=tl.float32)
        if HAS_VAC:
            e0 = m_f * (oq0 * oq0 + op0 * op0)
            e1 = m_f * (oq1 * oq1 + op1 * op1)
            lk = tl.load(kappa_ptr + offs_g, mask=mask_g, other=0.0).to(tl.float32)
            ksq = tl.minimum(tl.maximum(tl.exp(2.0 * lk), 1e-4), 1e4)
            denom = ksq + e0 + e1
            b0 = 2.0 * (e0 / denom)
            b1 = 2.0 * (e1 / denom)
            pvac = ksq / denom
        else:
            b0 = (2.0 * m_f) * (oq0 * oq0 + op0 * op0) - 1.0
            b1 = (2.0 * m_f) * (oq1 * oq1 + op1 * op1) - 1.0
            pvac = tl.full((BLOCK_G,), 1.0, dtype=tl.float32)

        base_obs = row * (3 * n_modes)
        tl.store(obs_out_ptr + base_obs + 2 * offs_g, oq0, mask=mask_g)
        tl.store(obs_out_ptr + base_obs + 2 * offs_g + 1, oq1, mask=mask_g)
        tl.store(obs_out_ptr + base_obs + n_modes + 2 * offs_g, op0, mask=mask_g)
        tl.store(obs_out_ptr + base_obs + n_modes + 2 * offs_g + 1, op1, mask=mask_g)
        tl.store(obs_out_ptr + base_obs + 2 * n_modes + 2 * offs_g, b0, mask=mask_g)
        tl.store(
            obs_out_ptr + base_obs + 2 * n_modes + 2 * offs_g + 1, b1, mask=mask_g
        )

        if WRITE_STATS:
            tl.store(q_obs_out_ptr + base_in + 2 * offs_g, oq0, mask=mask_g)
            tl.store(q_obs_out_ptr + base_in + 2 * offs_g + 1, oq1, mask=mask_g)
            tl.store(p_obs_out_ptr + base_in + 2 * offs_g, op0, mask=mask_g)
            tl.store(p_obs_out_ptr + base_in + 2 * offs_g + 1, op1, mask=mask_g)
            tl.store(born_out_ptr + base_in + 2 * offs_g, b0, mask=mask_g)
            tl.store(born_out_ptr + base_in + 2 * offs_g + 1, b1, mask=mask_g)
            tl.store(pvac_out_ptr + row * n_dyads + offs_g, pvac, mask=mask_g)


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


def _launch_triton_2d_osc_scan(
    a_cos: Tensor,
    a_sin: Tensor,
    u_q: Tensor,
    u_p: Tensor,
    a_nu: Tensor | None = None,
    u_nu: Tensor | None = None,
    reset_mask: Tensor | None = None,
    initial_state: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Launch fused 2D-oscillator (+ optional Lindblad nu) SRAM scan on [B, T, M]."""
    bsz, seq_len, n_modes = u_q.shape
    oq = torch.empty((bsz, seq_len, n_modes), dtype=torch.float32, device=u_q.device)
    op = torch.empty((bsz, seq_len, n_modes), dtype=torch.float32, device=u_q.device)
    has_nu = a_nu is not None and u_nu is not None
    onu = (
        torch.empty((bsz, seq_len, n_modes), dtype=torch.float32, device=u_q.device)
        if has_nu
        else oq
    )
    if seq_len == 0 or bsz == 0 or n_modes == 0:
        return oq, op, (onu if has_nu else None)

    c_c = a_cos.contiguous()
    s_c = a_sin.contiguous()
    uq_c = u_q.contiguous()
    up_c = u_p.contiguous()
    anu_c = a_nu.contiguous() if has_nu and a_nu is not None else c_c
    unu_c = u_nu.contiguous() if has_nu and u_nu is not None else uq_c

    has_reset = reset_mask is not None
    rst_c = reset_mask.contiguous() if has_reset and reset_mask is not None else c_c

    has_init_qp = bool(
        initial_state is not None
        and len(initial_state) >= 2
        and initial_state[0] is not None
        and initial_state[1] is not None
    )
    if has_init_qp and initial_state is not None:
        q0_t = initial_state[0]
        p0_t = initial_state[1]
        if q0_t.ndim == 3:
            q0_t = q0_t.squeeze(1)
            p0_t = p0_t.squeeze(1)
        q0_c = q0_t.contiguous()
        p0_c = p0_t.contiguous()
    else:
        q0_c = c_c
        p0_c = c_c

    has_init_nu = bool(
        has_nu
        and initial_state is not None
        and len(initial_state) >= 3
        and initial_state[2] is not None
    )
    if has_init_nu and initial_state is not None and initial_state[2] is not None:
        nu0_t = initial_state[2]
        if nu0_t.ndim == 3:
            nu0_t = nu0_t.squeeze(1)
        nu0_c = nu0_t.contiguous()
    else:
        nu0_c = c_c

    block_m = 8 if (bsz >= 8 and n_modes >= 64) else 4
    block_t = 64 if seq_len >= 64 else 32
    grid = (bsz, triton.cdiv(n_modes, block_m))
    _2d_osc_scan_fwd_kernel[grid](
        c_c,
        s_c,
        uq_c,
        up_c,
        anu_c,
        unu_c,
        rst_c,
        q0_c,
        p0_c,
        nu0_c,
        oq,
        op,
        onu,
        seq_len,
        n_modes,
        HAS_NU=has_nu,
        HAS_RESET=has_reset,
        HAS_INIT_QP=has_init_qp,
        HAS_INIT_NU=has_init_nu,
        BLOCK_T=block_t,
        BLOCK_M=block_m,
        num_warps=4 if (block_t * block_m) >= 256 else 2,
    )
    return oq, op, (onu if has_nu else None)


def segmented_2d_osc_scan(
    a_cos: Tensor,
    a_sin: Tensor,
    u_q: Tensor,
    u_p: Tensor,
    a_nu: Tensor | None = None,
    u_nu: Tensor | None = None,
    reset_mask: Tensor | None = None,
    initial_state: tuple[Tensor, ...] | None = None,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Fused FP32 segmented 2D damped oscillator (+ Lindblad nu) scan on [B, T, M]."""
    need_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (a_cos, a_sin, u_q, u_p, a_nu, u_nu)
    )
    if _can_run_triton(u_q, use_triton) and not need_grad:
        return _launch_triton_2d_osc_scan(
            a_cos=a_cos,
            a_sin=a_sin,
            u_q=u_q,
            u_p=u_p,
            a_nu=a_nu,
            u_nu=u_nu,
            reset_mask=reset_mask,
            initial_state=initial_state,
        )

    if _can_run_triton(u_q, use_triton) and need_grad:
        c_f = a_cos.float()
        s_f = a_sin.float()
        anu_f = a_nu.float() if a_nu is not None else None
        if reset_mask is not None:
            keep = (~reset_mask).to(torch.float32).unsqueeze(-1)
            c_f = c_f * keep
            s_f = s_f * keep
            if anu_f is not None:
                anu_f = anu_f * keep
        c_init = (
            (initial_state[0], initial_state[1])
            if (
                initial_state is not None
                and len(initial_state) >= 2
                and initial_state[0] is not None
                and initial_state[1] is not None
            )
            else None
        )
        nu_init = (
            initial_state[2]
            if (
                initial_state is not None
                and len(initial_state) >= 3
                and initial_state[2] is not None
            )
            else None
        )
        q_seq, p_seq = segmented_complex_scan(
            c_f, s_f, u_q, u_p, initial_state=c_init, use_triton=True
        )
        nu_seq = (
            segmented_scalar_scan(
                anu_f, u_nu, initial_state=nu_init, use_triton=True
            )
            if (anu_f is not None and u_nu is not None)
            else None
        )
        return q_seq, p_seq, nu_seq

    return pytorch_segmented_2d_osc_scan(
        a_cos=a_cos,
        a_sin=a_sin,
        u_q=u_q,
        u_p=u_p,
        a_nu=a_nu,
        u_nu=u_nu,
        reset_mask=reset_mask,
        initial_state=initial_state,
    )


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
    use_triton: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    """Advance (q, p[, nu]) across S_max steps up to valid_lens in FP32."""
    bsz, s_max, n_modes = u_q.shape
    has_nu = nu_init is not None and a_nu is not None and u_nu is not None

    if _can_run_triton(u_q, use_triton) and s_max > 0 and bsz > 0 and n_modes > 0:
        c_c = a_cos.contiguous()
        s_c = a_sin.contiguous()
        uq_c = u_q.contiguous()
        up_c = u_p.contiguous()
        qi_c = q_init.contiguous()
        pi_c = p_init.contiguous()
        vl_c = valid_lens.to(device=u_q.device).contiguous()

        q_seq = torch.empty(
            (bsz, s_max, n_modes), dtype=torch.float32, device=u_q.device
        )
        p_seq = torch.empty(
            (bsz, s_max, n_modes), dtype=torch.float32, device=u_q.device
        )
        q_fin = torch.empty((bsz, n_modes), dtype=torch.float32, device=u_q.device)
        p_fin = torch.empty((bsz, n_modes), dtype=torch.float32, device=u_q.device)

        if has_nu and a_nu is not None and u_nu is not None and nu_init is not None:
            anu_c = a_nu.contiguous()
            unu_c = u_nu.contiguous()
            nui_c = nu_init.contiguous()
            nu_seq = torch.empty(
                (bsz, s_max, n_modes), dtype=torch.float32, device=u_q.device
            )
            nu_fin = torch.empty(
                (bsz, n_modes), dtype=torch.float32, device=u_q.device
            )
        else:
            anu_c = c_c
            unu_c = uq_c
            nui_c = qi_c
            nu_seq = q_seq
            nu_fin = q_fin

        block_m = min(256, triton.next_power_of_2(max(16, n_modes)))
        grid = (bsz, triton.cdiv(n_modes, block_m))
        _multistep_2d_osc_decode_kernel[grid](
            c_c,
            s_c,
            uq_c,
            up_c,
            qi_c,
            pi_c,
            vl_c,
            anu_c,
            unu_c,
            nui_c,
            q_seq,
            p_seq,
            q_fin,
            p_fin,
            nu_seq,
            nu_fin,
            s_max,
            n_modes,
            HAS_NU=has_nu,
            BLOCK_M=block_m,
            num_warps=2 if block_m <= 128 else 4,
        )
        return (
            q_seq,
            p_seq,
            q_fin,
            p_fin,
            (nu_seq if has_nu else None),
            (nu_fin if has_nu else None),
        )

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


def fused_2d_osc_dynamics_multistep_decode(
    r_q: Tensor,  # [B, S_max, M]
    r_p: Tensor,  # [B, S_max, M]
    raw_dt: Tensor,  # [B, S_max, M]
    raw_env: Tensor | None,  # [B, S_max, M]
    log_eta: Tensor,  # [M]
    kvn_omega: Tensor,  # [M]
    in_gain_q: Tensor,  # [M]
    in_gain_p: Tensor,  # [M]
    lindblad_gain: Tensor | None,  # [M]
    q_init: Tensor,  # [B, M]
    p_init: Tensor,  # [B, M]
    nu_init: Tensor | None,  # [B, M]
    valid_lens: Tensor,  # [B]
    damping_min: float,
    damping_max: float,
    state_dtype: torch.dtype = torch.float32,
    dtheta_dtype: torch.dtype = torch.bfloat16,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]:
    """Single-kernel 2D oscillator + Lindblad dynamics & multi-step decode."""
    bsz, s_max, n_modes = r_q.shape
    has_nu = (
        raw_env is not None and lindblad_gain is not None and nu_init is not None
    )
    if _can_run_triton(r_q, use_triton) and s_max > 0 and bsz > 0 and n_modes > 0:
        rq_c = r_q.contiguous()
        rp_c = r_p.contiguous()
        rdt_c = raw_dt.contiguous()
        qi_c = q_init.contiguous()
        pi_c = p_init.contiguous()
        vl_c = valid_lens.to(device=r_q.device).contiguous()

        q_seq = torch.empty(
            (bsz, s_max, n_modes), dtype=torch.float32, device=r_q.device
        )
        p_seq = torch.empty(
            (bsz, s_max, n_modes), dtype=torch.float32, device=r_q.device
        )
        q_fin = torch.empty((bsz, n_modes), dtype=state_dtype, device=r_q.device)
        p_fin = torch.empty((bsz, n_modes), dtype=state_dtype, device=r_q.device)
        dtheta_out = torch.empty(
            (bsz, s_max, n_modes), dtype=dtheta_dtype, device=r_q.device
        )

        if (
            has_nu
            and raw_env is not None
            and lindblad_gain is not None
            and nu_init is not None
        ):
            renv_c = raw_env.contiguous()
            lg_c = lindblad_gain.contiguous()
            nui_c = nu_init.contiguous()
            nu_fin = torch.empty((bsz, n_modes), dtype=state_dtype, device=r_q.device)
        else:
            renv_c = rdt_c
            lg_c = log_eta.contiguous()
            nui_c = qi_c
            nu_fin = q_fin

        block_m = min(256, triton.next_power_of_2(max(16, n_modes)))
        grid = (bsz, triton.cdiv(n_modes, block_m))
        _fused_2d_osc_dynamics_decode_kernel[grid](
            rq_c,
            rp_c,
            rdt_c,
            renv_c,
            log_eta.contiguous(),
            kvn_omega.contiguous(),
            in_gain_q.contiguous(),
            in_gain_p.contiguous(),
            lg_c,
            qi_c,
            pi_c,
            nui_c,
            vl_c,
            q_seq,
            p_seq,
            q_fin,
            p_fin,
            nu_fin,
            dtheta_out,
            s_max,
            n_modes,
            float(damping_min),
            float(damping_max),
            HAS_NU=has_nu,
            ROUND_DT_BF16=(raw_dt.dtype == torch.bfloat16),
            ROUND_DT_FP16=(raw_dt.dtype == torch.float16),
            BLOCK_M=block_m,
            num_warps=2 if block_m <= 128 else 4,
        )
        return (
            q_seq,
            p_seq,
            q_fin,
            p_fin,
            (nu_fin if has_nu else None),
            dtheta_out,
        )

    dt_f = torch.nn.functional.softplus(raw_dt).float()
    eta = (
        torch.exp(log_eta.float())
        .clamp(min=damping_min, max=damping_max)
        .view(1, 1, -1)
    )
    omega_pole = kvn_omega.float().view(1, 1, -1)
    dtheta_f = dt_f * omega_pole
    decay = torch.exp(-eta * dt_f)
    a_cos = decay * torch.cos(dtheta_f)
    a_sin = decay * torch.sin(dtheta_f)
    u_q_f = dt_f * in_gain_q.float().view(1, 1, -1) * r_q.float()
    u_p_f = dt_f * in_gain_p.float().view(1, 1, -1) * r_p.float()
    if has_nu and raw_env is not None and lindblad_gain is not None:
        a_nu = torch.exp(-2.0 * eta * dt_f)
        g_sq = lindblad_gain.float().square().view(1, 1, -1)
        u_nu = dt_f * g_sq * torch.nn.functional.softplus(raw_env.float())
    else:
        a_nu, u_nu = None, None
    q_seq, p_seq, q_fin_f, p_fin_f, _, nu_fin_f = fused_multistep_complex_decode(
        a_cos=a_cos,
        a_sin=a_sin,
        u_q=u_q_f,
        u_p=u_p_f,
        q_init=q_init.float(),
        p_init=p_init.float(),
        valid_lens=valid_lens,
        a_nu=a_nu,
        u_nu=u_nu,
        nu_init=nu_init.float() if nu_init is not None else None,
        use_triton=False,
    )
    return (
        q_seq,
        p_seq,
        q_fin_f.to(state_dtype),
        p_fin_f.to(state_dtype),
        nu_fin_f.to(state_dtype) if nu_fin_f is not None else None,
        dtheta_f.to(dtheta_dtype),
    )


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

    if _can_run_triton(u_f, use_triton) and s_max > 0 and bsz > 0 and n_dyads > 0:
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
        n_modes = nu_init.size(-1)
        if _can_run_triton(u_nu, use_triton) and s_max > 0 and bsz > 0 and n_modes > 0:
            anu_c = a_nu.contiguous()
            unu_c = u_nu.contiguous()
            nui_c = nu_init.contiguous()
            nu_seq = torch.empty(
                (bsz, s_max, n_modes), dtype=torch.float32, device=u_nu.device
            )
            nu_final = torch.empty(
                (bsz, n_modes), dtype=torch.float32, device=u_nu.device
            )
            block_m = min(256, triton.next_power_of_2(max(16, n_modes)))
            grid = (bsz, triton.cdiv(n_modes, block_m))
            _multistep_scalar_decode_kernel[grid](
                anu_c,
                unu_c,
                nui_c,
                v_lens,
                nu_seq,
                nu_final,
                s_max,
                n_modes,
                BLOCK_M=block_m,
                num_warps=2 if block_m <= 128 else 4,
            )
        else:
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


def fused_engram_hash_rmsnorm(
    p1: Tensor,
    p2: Tensor,
    p3: Tensor,
    table_weight: Tensor,  # [buckets, engram_dim]
    rms_weight: Tensor,  # [engram_dim]
    buckets: int,
    eps: float = 1e-6,
    use_triton: bool = True,
) -> Tensor:
    """Single-kernel 4-head prime hash lookup + average + RMSNorm."""
    orig_shape = p1.shape
    engram_dim = table_weight.size(1)
    n_rows = p1.numel()
    if (
        _can_run_triton(table_weight, use_triton)
        and not torch.is_grad_enabled()
        and n_rows > 0
        and engram_dim > 0
    ):
        p1_c = p1.contiguous().view(-1)
        p2_c = p2.contiguous().view(-1)
        p3_c = p3.contiguous().view(-1)
        out = torch.empty(
            (n_rows, engram_dim),
            dtype=table_weight.dtype,
            device=table_weight.device,
        )
        block_d = triton.next_power_of_2(max(16, engram_dim))
        _fused_engram_hash_rmsnorm_kernel[(n_rows,)](
            p1_c,
            p2_c,
            p3_c,
            table_weight.contiguous(),
            rms_weight.contiguous(),
            out,
            int(buckets),
            int(engram_dim),
            float(eps),
            BLOCK_D=block_d,
            num_warps=2 if block_d <= 128 else 4,
        )
        return out.view(*orig_shape, engram_dim)

    p1_l = p1.long().clamp_min(0)
    p2_l = p2.long().clamp_min(0)
    p3_l = p3.long().clamp_min(0)
    h0 = torch.remainder(p2_l * 1315423911 + p1_l * 2654435761 + 0x9E3779B9, buckets)
    h1 = torch.remainder(p2_l * 2246822519 + p1_l * 3266489917 + 0x85EBCA6B, buckets)
    h2 = torch.remainder(
        p3_l * 668265263 + p2_l * 374761393 + p1_l * 2654435761 + 0xC2B2AE35, buckets
    )
    h3 = torch.remainder(
        p3_l * 374761393 + p2_l * 1315423911 + p1_l * 668265263 + 0x27D4EB2F, buckets
    )
    emb = 0.25 * (
        torch.nn.functional.embedding(h0, table_weight)
        + torch.nn.functional.embedding(h1, table_weight)
        + torch.nn.functional.embedding(h2, table_weight)
        + torch.nn.functional.embedding(h3, table_weight)
    )
    return torch.nn.functional.rms_norm(emb, (engram_dim,), rms_weight, eps)


def fused_interferometer_born_obs(
    q_f: Tensor,
    p_f: Tensor,
    out_dtype: torch.dtype,
    dyad_theta: Tensor | None = None,
    dyad_phase: Tensor | None = None,
    vacuum_log_kappa: Tensor | None = None,
    return_stats: bool = False,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """Single-kernel dyadic interferometer + vacuum-Born observables + cat."""
    orig_shape = q_f.shape
    n_modes = orig_shape[-1]
    n_rows = q_f.numel() // max(1, n_modes)
    has_interf = dyad_theta is not None and dyad_phase is not None
    has_vac = vacuum_log_kappa is not None and vacuum_log_kappa.numel() == (
        n_modes // 2
    )

    if (
        _can_run_triton(q_f, use_triton)
        and not torch.is_grad_enabled()
        and n_modes > 0
        and n_modes % 2 == 0
        and n_rows > 0
    ):
        n_dyads = n_modes // 2
        q_c = q_f.contiguous().view(n_rows, n_modes)
        p_c = p_f.contiguous().view(n_rows, n_modes)
        obs_out = torch.empty(
            (n_rows, 3 * n_modes), dtype=out_dtype, device=q_f.device
        )

        th_c = dyad_theta.contiguous() if has_interf and dyad_theta is not None else q_c
        ph_c = dyad_phase.contiguous() if has_interf and dyad_phase is not None else q_c
        lk_c = (
            vacuum_log_kappa.contiguous()
            if has_vac and vacuum_log_kappa is not None
            else q_c
        )

        if return_stats:
            q_obs_out = torch.empty(
                (n_rows, n_modes), dtype=torch.float32, device=q_f.device
            )
            p_obs_out = torch.empty(
                (n_rows, n_modes), dtype=torch.float32, device=q_f.device
            )
            born_out = torch.empty(
                (n_rows, n_modes), dtype=out_dtype, device=q_f.device
            )
            pvac_out = torch.empty(
                (n_rows, n_dyads), dtype=torch.float32, device=q_f.device
            )
        else:
            q_obs_out = q_c
            p_obs_out = p_c
            born_out = obs_out
            pvac_out = q_c

        block_g = triton.next_power_of_2(max(16, n_dyads))
        _fused_interferometer_born_obs_kernel[(n_rows,)](
            q_c,
            p_c,
            th_c,
            ph_c,
            lk_c,
            obs_out,
            q_obs_out,
            p_obs_out,
            born_out,
            pvac_out,
            n_dyads,
            HAS_INTERF=has_interf,
            HAS_VAC=has_vac,
            WRITE_STATS=return_stats,
            BLOCK_G=block_g,
            num_warps=2 if block_g <= 128 else 4,
        )
        obs_view = obs_out.view(*orig_shape[:-1], 3 * n_modes)
        if return_stats:
            return (
                obs_view,
                q_obs_out.view(orig_shape),
                p_obs_out.view(orig_shape),
                born_out.view(orig_shape),
                pvac_out.view(*orig_shape[:-1], n_dyads),
            )
        return obs_view, None, None, None, None

    raise RuntimeError("Fallback handled by caller.")


if _HAS_TRITON:

    @triton.jit
    def _fused_rmsnorm_trig_mod_kernel(
        z_ptr,
        add_ptr,
        w_ptr,
        zbar_ptr,
        hmod_ptr,
        D: tl.constexpr,
        EPS: tl.constexpr,
        HAS_ADD: tl.constexpr,
        WRITE_HMOD: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        base = row * D + offs
        z_val = tl.load(z_ptr + base, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD:
            z_val = z_val + tl.load(add_ptr + base, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(z_val * z_val, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + EPS)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        z_bar_f = (z_val * rstd * w).to(zbar_ptr.dtype.element_ty).to(tl.float32)
        tl.store(zbar_ptr + base, z_bar_f.to(zbar_ptr.dtype.element_ty), mask=mask)
        if WRITE_HMOD:
            h_f = 0.7071067811865475 * (tl.sin(z_bar_f) + tl.cos(z_bar_f))
            tl.store(hmod_ptr + base, h_f.to(hmod_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _fused_dyadic_rff_2stage_trig_kernel(
        p0_ptr,
        p1_ptr,
        ph_ptr,
        alpha_ptr,
        rq_ptr,
        rp_ptr,
        M: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_M)
        mask = offs < M
        base = row * M + offs

        pr0 = tl.load(p0_ptr + base, mask=mask, other=0.0).to(tl.float32)
        pr1 = tl.load(p1_ptr + base, mask=mask, other=0.0).to(tl.float32)
        ph0 = tl.load(ph_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ph1 = tl.load(ph_ptr + M + offs, mask=mask, other=0.0).to(tl.float32)
        raw_a = tl.load(alpha_ptr + offs, mask=mask, other=0.0).to(
            rq_ptr.dtype.element_ty
        ).to(tl.float32)

        phi0 = pr0 + ph0
        phi1 = 2.0 * pr1 + ph1
        c0 = tl.cos(phi0)
        s0 = tl.sin(phi0)
        c1 = tl.cos(phi1)
        s1 = tl.sin(phi1)
        c_acc = (c0 * c1 - s0 * s1).to(rq_ptr.dtype.element_ty).to(tl.float32)
        s_acc = (c0 * s1 + s0 * c1).to(rq_ptr.dtype.element_ty).to(tl.float32)

        # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) with clamp for overflow safety
        a_clamped = tl.minimum(tl.maximum(raw_a, -15.0), 15.0)
        e2a = tl.exp(2.0 * a_clamped)
        alpha = ((e2a - 1.0) / (e2a + 1.0)).to(rq_ptr.dtype.element_ty).to(tl.float32)

        rq = SCALE * (c_acc + alpha * (c_acc * c_acc - s_acc * s_acc))
        rp = SCALE * (s_acc + 2.0 * alpha * (c_acc * s_acc))
        tl.store(rq_ptr + base, rq.to(rq_ptr.dtype.element_ty), mask=mask)
        tl.store(rp_ptr + base, rp.to(rp_ptr.dtype.element_ty), mask=mask)


def fused_rmsnorm_trig_mod(
    z: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    add_residual: Tensor | None = None,
    write_hmod: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Fused RMSNorm (+ optional pre-add + optional sin/cos h_mod) for dyadic RFF."""
    orig_shape = z.shape
    dim = orig_shape[-1]
    z_c = z.contiguous().view(-1, dim)
    n_rows = z_c.size(0)
    zbar_out = torch.empty_like(z_c)
    hmod_out = torch.empty_like(z_c) if write_hmod else zbar_out
    if n_rows == 0:
        return zbar_out.view(orig_shape), (
            hmod_out.view(orig_shape) if write_hmod else None
        )
    add_c = (
        add_residual.contiguous().view(-1, dim) if add_residual is not None else z_c
    )
    block_d = triton.next_power_of_2(max(32, dim))
    _fused_rmsnorm_trig_mod_kernel[(n_rows,)](
        z_c,
        add_c,
        weight.contiguous(),
        zbar_out,
        hmod_out,
        D=dim,
        EPS=float(eps),
        HAS_ADD=add_residual is not None,
        WRITE_HMOD=write_hmod,
        BLOCK_D=block_d,
        num_warps=4 if block_d >= 512 else 2,
    )
    return zbar_out.view(orig_shape), (
        hmod_out.view(orig_shape) if write_hmod else None
    )


def fused_dyadic_rff_2stage_trig(
    proj0: Tensor,
    proj1: Tensor,
    phase_rff: Tensor,
    raw_alpha: Tensor,
    out_dtype: torch.dtype,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Fused 2-stage dyadic RFF trig combine + 2nd-harmonic quadrature readout."""
    orig_shape = proj0.shape
    n_modes = orig_shape[-1]
    p0_c = proj0.contiguous().view(-1, n_modes)
    p1_c = proj1.contiguous().view(-1, n_modes)
    n_rows = p0_c.size(0)
    rq_out = torch.empty((n_rows, n_modes), dtype=out_dtype, device=proj0.device)
    rp_out = torch.empty((n_rows, n_modes), dtype=out_dtype, device=proj0.device)
    if n_rows == 0:
        return rq_out.view(orig_shape), rp_out.view(orig_shape)
    block_m = triton.next_power_of_2(max(32, n_modes))
    _fused_dyadic_rff_2stage_trig_kernel[(n_rows,)](
        p0_c,
        p1_c,
        phase_rff.contiguous(),
        raw_alpha.contiguous(),
        rq_out,
        rp_out,
        M=n_modes,
        SCALE=float(scale),
        BLOCK_M=block_m,
        num_warps=2 if block_m <= 256 else 4,
    )
    return rq_out.view(orig_shape), rp_out.view(orig_shape)

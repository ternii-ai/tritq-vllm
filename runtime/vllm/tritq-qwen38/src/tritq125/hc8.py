# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""hc8.py: the hyper-connection (HC) projections at 8 bits.
Format = Q8_0: per 32 consecutive inputs of a row, key = fp16(absmax / 127)
(a zero key -> 1), codes = round-half-even(x / key) clamped to +-127. Quantized at load time from the checkpoint's bf16 HC weights
(deterministic; equal to an offline write of the same rule, checked bitwise by a unit test).
Kernel layout (N contiguous, as the trit kernels): q [K, N] int8, s [K/32, N] fp16. y[m, n] = sum_g s[g, n] * (x[m, g*32:(g+1)*32] . q[g, n])
with the 32-group dot in fp32 (tl.dot, codes exact in bf16) - i.e. the exact weights q * key. Split-K partials (few output tiles: the down / inject projection has
N = 336) are reduced in the kernel by the last-arriving program of each tile in a fixed order (glue._red4_store), so the result is
deterministic and graph-replay safe (the last program resets its counter)."""
import torch
import triton
import triton.language as tl
from .glue import _counters, _red4_store

GQ = 32


def quantize(w: torch.Tensor):
    """w [N, K] (bf16) -> (q [K, N] int8, s [K/32, N] fp16) (dequant q * s == the Q8_0 rule's values)."""
    N, K = w.shape
    assert K % GQ == 0, (N, K)
    x = w.float().reshape(N, K // GQ, GQ)
    s = (x.abs().amax(-1) / 127).half()
    sf = s.float()
    sf = torch.where(sf == 0, torch.ones_like(sf), sf)
    q = (x / sf[..., None]).round().clamp(-127, 127).to(torch.int8)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return q.reshape(N, K).t().contiguous(), s.t().contiguous()


def dequant(q, s):
    """-> fp32 weight [N, K] = q * key (exact)."""
    K, N = q.shape
    return (q.float().reshape(K // GQ, GQ, N) * s.float()[:, None, :]).reshape(K, N).t()


@triton.jit
def _q8_kernel(x_ptr, q_ptr, s_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, NG, stride_xm, NTN,
               SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr,
               SILU_N: tl.constexpr, HC: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < N
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64); m_mask = offs_m < M
    offs_g = tl.arange(0, 32)
    gper = tl.cdiv(NG, SPLIT_K); g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(g0, g1, GPI):                     # GPI groups per iteration (unrolled; more loads in flight), same per-group order
        for u in tl.static_range(GPI):
            gg = g + u
            ok = gg < g1
            kk = gg * 32 + offs_g
            a = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None] & ok, other=0.0)
            w = tl.load(q_ptr + kk[:, None].to(tl.int64) * N + offs_n[None, :], mask=n_mask[None, :] & ok, other=0).to(tl.bfloat16)
            sc = tl.load(s_ptr + gg * N + offs_n, mask=n_mask & ok, other=0.0).to(tl.float32)
            part = tl.dot(a, w)
            acc += part * sc[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    off2d = offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + off2d, _epi(acc, offs_n, SILU_N, HC), mask=mask)
    else:
        tl.store(ws_ptr + pid_k * M * N + off2d, acc, mask=mask)
        tile = pid_m * NTN + pid_n
        old = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if old == SPLIT_K - 1:
            r = _red4_store(ws_ptr, M * N, off2d, mask, SPLIT_K, BLOCK_M, BLOCK_N)
            tl.store(out_ptr + off2d, _epi(r, offs_n, SILU_N, HC), mask=mask)
            tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit
def _epi(acc, offs_n, SILU_N: tl.constexpr, HC: tl.constexpr):
    """bf16 rounding; columns < SILU_N then get vLLM's hc_silu exactly (ops/hc.py _hc_silu_kernel: x = bf16 value / HC; x * sigmoid(x))"""
    y = acc.to(tl.bfloat16)
    if SILU_N > 0:
        v = y.to(tl.float32) / HC
        v = v * tl.sigmoid(v)
        y = tl.where((offs_n < SILU_N)[None, :], v.to(tl.bfloat16), y)
    return y


def _cfg(M, N, NG, bn=None, target=192, max_split=64):
    BM = 16 if M <= 16 else 32 if M <= 32 else 64
    BN = bn or 64
    ntn, ntm = triton.cdiv(N, BN), triton.cdiv(M, BM)
    split = 1
    if ntm <= 64:
        while ntn * ntm * split < target and split * 2 <= min(NG, max_split) and NG % (split * 2) == 0:
            split *= 2
    return BM, BN, split


# decode configs tuned per shape (tuned; M <= 16): (N, K) -> (BLOCK_N, split, warps, stages, GPI)
TUNED = {(336, 10240): (128, 16, 8, 3, 2),        # down + inject (+ 12 pad rows): 21.7 us vs 28.4 us bf16 CuTe fused (M = 1)
         (10240, 320): (128, 1, 8, 2, 2),          # up: 17.9 us vs 26.4 us bf16 SM121 skinny plan
         (320, 10240): (64, 16, 4, 3, 2)}          # final mixer down: 18.8 us vs 33.9 us bf16 cuBLAS


def q8_linear(x, q, s, bn=None, split=None, warps=None, stages=None, gpi=None, silu_n=0, hc=1):
    """x [M, K] bf16 (row stride free, unit column stride), q [K, N] int8, s [K/32, N] fp16 -> [M, N] bf16.
    silu_n > 0: columns < silu_n get hc_silu(., hc) fused (the HC down / inject projection: lora columns)."""
    assert x.stride(1) == 1
    M, K = x.shape; N = q.shape[1]; NG = K // GQ
    assert q.shape[0] == K and s.shape == (NG, N), (x.shape, q.shape, s.shape)
    t = TUNED.get((N, K)) if M <= 16 else None
    if t is not None and bn is None and split is None:
        bn, split, warps, stages, gpi = t
    BM, BN, sk = _cfg(M, N, NG, bn)
    if split is not None: sk = split
    warps, stages, gpi = warps or 4, stages or 3, gpi or 1
    ntn, ntm = triton.cdiv(N, BN), triton.cdiv(M, BM)
    out = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if sk > 1:
        ws = torch.empty(sk, M, N, device=x.device, dtype=torch.float32)
        cnt = _counters(("hc8", q.data_ptr(), BN), ntn * 64)
    else:
        ws, cnt = out, out
    _q8_kernel[(ntn, ntm, sk)](x, q, s, out, ws, cnt, M, N, NG, x.stride(0), ntn, SPLIT_K=sk, BLOCK_M=BM, BLOCK_N=BN, GPI=gpi,
                               SILU_N=silu_n, HC=hc, num_warps=warps, num_stages=stages)
    return out

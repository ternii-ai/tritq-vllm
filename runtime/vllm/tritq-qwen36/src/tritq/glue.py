# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""glue.py: decode-time glue removal around the trit kernels in kern.py. Same per-element arithmetic as kern.py (the trit
_tile_acc, BLOCK_M 16, BLOCK_N 64, the same split-K group ranges), so results are BIT-IDENTICAL to the kern.py path wherever the
reduction order is reproduced; that order was measured (by a probe script): torch's fp32 [sk, ...].sum(0) uses 4 interleaved
accumulators acc[i % 4] combined as ((a0 + a1) + a2) + a3; the bf16 top-k .sum(1) matched every order tried (sums of 8-10 bf16 values are
exact in fp32 in practice); F.silu == x / (1 + exp(-x)) in fp32 bitwise (libdevice expf + div_rn here).
  L2  tq_linear_rk: split-K partials reduced IN the kernel by the last-arriving program of each output tile (atomic counter, fixed order
      above, bf16 cast fused) - replaces torch.sum(0) + .to(bf16).
  L1  tq_moe_decode (T * top_k <= 64): no moe_align / sort; w13 per (slot, tile pair gate|up) with the split-K reduce, bf16 rounding and
      SiLU * mul fused; w2 loops over the token's top-k experts, router-weight multiply, bf16 rounding and the top-k sum in registers.
  L3  bf16_gemv: one Triton GEMV for the unquantized bf16 linears (routers, shared_expert_gate, in_proj_ba) at M <= 8 (ORDER-CHANGING vs
      cuBLAS: its own fixed order).
MASK (as in the Qwen3.8-Flash-Next plugin): L1 used to load the expert id
unchecked; now an invalid id (< 0 or >= E) is MASKED - that slot's activations load as 0, its weight loads use expert 0 only to stay in
bounds, and its act / w2 value is forced to exactly 0 (= router weight 0). No-op for valid ids (bit-identical, unit test).
Counters: one int32 buffer per weight (keyed by its data pointer), zeroed once outside graph capture; the last program of a tile resets
its counter, so graph replays need no fill kernel."""
import os
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .kern import _tile_acc, _cfg, G

_CNT = {}


def _counters(tag, n):
    b = _CNT.get(tag)
    if b is None or b.numel() < n:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("glue counters must be allocated before cudagraph capture")
        b = torch.zeros(max(n, 1024), dtype=torch.int32, device="cuda")
        _CNT[tag] = b
    return b


@triton.jit
def _red4_store(ws_ptr, stride_s, off2d, mask, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """fixed-order sum of the SPLIT_K partials at ws[k * stride_s + off2d] = torch's [sk,...].sum(0) order (SPLIT_K 1: the value itself)"""
    if SPLIT_K == 1:
        return tl.load(ws_ptr + off2d, mask=mask, other=0.0, cache_modifier=".cg")
    a0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32); a1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32); a3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        p = tl.load(ws_ptr + k * stride_s + off2d, mask=mask, other=0.0, cache_modifier=".cg")
        if k % 4 == 0:
            a0 += p
        elif k % 4 == 1:
            a1 += p
        elif k % 4 == 2:
            a2 += p
        else:
            a3 += p
    return ((a0 + a1) + a2) + a3


@triton.jit
def _dense_rk(a_ptr, key_ptr, lo_ptr, hi_ptr, out_ptr, ws_ptr, cnt_ptr, M, NOUT, NG, stride_am, LD, LDH, HCOL0, SEGB, NTN,
              NLEV: tl.constexpr, WB: tl.constexpr, NLEV2: tl.constexpr, WB2: tl.constexpr, TWO: tl.constexpr,
              SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < NOUT
    offs_w = tl.arange(0, 32); w_mask = offs_w < 26
    arow = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64); m_mask = arow < M
    gper = tl.cdiv(NG, SPLIT_K); g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    if TWO:
        if pid_n * BLOCK_N >= SEGB:
            acc = _tile_acc(a_ptr, arow, m_mask, key_ptr, lo_ptr, hi_ptr, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                            LD, 0, LDH, HCOL0, NLEV2, WB2, BLOCK_M, BLOCK_N)
        else:
            acc = _tile_acc(a_ptr, arow, m_mask, key_ptr, lo_ptr, hi_ptr, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                            LD, 0, LDH, HCOL0, NLEV, WB, BLOCK_M, BLOCK_N)
    else:
        acc = _tile_acc(a_ptr, arow, m_mask, key_ptr, lo_ptr, hi_ptr, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                        LD, 0, LDH, HCOL0, NLEV, WB, BLOCK_M, BLOCK_N)
    mask = m_mask[:, None] & n_mask[None, :]
    off2d = arow[:, None] * LD + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + off2d, acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(ws_ptr + pid_k * M * LD + off2d, acc, mask=mask)
        tile = pid_m * NTN + pid_n
        old = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if old == SPLIT_K - 1:
            s = _red4_store(ws_ptr, M * LD, off2d, mask, SPLIT_K, BLOCK_M, BLOCK_N)
            tl.store(out_ptr + off2d, s.to(out_ptr.dtype.element_ty), mask=mask)
            tl.atomic_xchg(cnt_ptr + tile, 0)


def tq_linear_rk(x, key, lo, hi, nlev, segs=None):
    """= kern.tq_linear (same tiles / split / per-part levels for <= 2 segments), split-K reduced in the kernel."""
    from .kern import tq_linear
    M, K = x.shape; NG, LD = key.shape; assert K == NG * G
    if segs is not None and len(segs) > 2:
        return tq_linear(x, key, lo, hi, nlev, segs=segs)
    if segs is None or len(segs) == 1:
        nl = nlev if segs is None else segs[0][2]
        two, segb, nl2, hshift = False, LD, nl, 0
        if segs is not None and nl == 27: hshift = segs[0][3] - segs[0][0]
    else:
        (c0, n0, nl, h0), (c1, n1, nl2, h1) = segs
        two, segb = True, c1
        hshift = (h0 - c0) if nl == 27 else (h1 - c1)
    BM, BN, sk = _cfg(M, LD, NG)
    WB, WB2 = (16 if nl == 9 else 24), (16 if nl2 == 9 else 24)
    hp = hi if hi is not None else lo
    LDH = hi.shape[-1] if hi is not None else LD
    out = torch.empty(M, LD, device=x.device, dtype=torch.bfloat16)
    ntn, ntm = triton.cdiv(LD, BN), triton.cdiv(M, BM)
    if sk > 1:
        ws = torch.empty(sk, M, LD, device=x.device, dtype=torch.float32)
        if ntm > 64: return tq_linear(x, key, lo, hi, nlev, segs=segs)
        cnt = _counters(("d", key.data_ptr()), ntn * 64)
    else:
        ws, cnt = out, out
    _dense_rk[(ntn, ntm, sk)](x, key, lo, hp, out, ws, cnt, M, LD, NG, x.stride(0), LD, LDH, hshift, segb, ntn,
                              NLEV=nl, WB=WB, NLEV2=nl2, WB2=WB2, TWO=two, SPLIT_K=sk, BLOCK_M=BM, BLOCK_N=BN, num_warps=4, num_stages=2)
    return out


@triton.jit
def _moe13_dec(x_ptr, ids_ptr, key_ptr, lo_ptr, hi_ptr, act_ptr, ws_ptr, cnt_ptr, NG, I, NV, stride_xm, stride_e_key, stride_e_lo,
               NTN, NE, TOPK: tl.constexpr, NLEV: tl.constexpr, WB: tl.constexpr, SPLIT_K: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """grid (2 * NTN, NV, SPLIT_K): one slot (token, k) x one gate OR up tile x one K split (as many programs as kern.tq_moe's w13).
    Partials -> fp32 workspace; the last of the 2 * SPLIT_K programs of a (slot, tile) reduces gate and up in torch's fixed order,
    rounds to bf16 (= the w13 output store), SiLU(gate) * up in fp32 -> act bf16."""
    pid_n = tl.program_id(0); s = tl.program_id(1).to(tl.int64); pid_k = tl.program_id(2)
    half = pid_n // NTN; j = pid_n % NTN
    eid = tl.load(ids_ptr + s).to(tl.int64)
    valid = (eid >= 0) & (eid < NE)                                  # MASK: invalid id -> weight 0 (no clamp)
    e = tl.where(valid, eid, 0)                                      # in-bounds weight loads only
    rr = tl.arange(0, BLOCK_M); m_mask = rr == 0
    am_mask = m_mask & valid                                         # invalid slot: activations load as 0
    arow = tl.zeros((BLOCK_M,), dtype=tl.int64) + s // TOPK
    key_base = key_ptr + e * stride_e_key; lo_base = lo_ptr + e * stride_e_lo; hi_base = hi_ptr + e * stride_e_lo
    offs_w = tl.arange(0, 32); w_mask = offs_w < 26
    offs_g = j * BLOCK_N + tl.arange(0, BLOCK_N); g_mask = offs_g < I
    offs_c = offs_g + half * I
    LD = 2 * I
    gper = tl.cdiv(NG, SPLIT_K); g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    acc = _tile_acc(x_ptr, arow, am_mask, key_base, lo_base, hi_base, offs_c, g_mask, offs_w, w_mask, stride_xm, g0, g1,
                    LD, 0, LD, 0, NLEV, WB, BLOCK_M, BLOCK_N)
    mask = m_mask[:, None] & g_mask[None, :]
    srow = s + rr * 0
    tl.store(ws_ptr + pid_k * NV * LD + srow[:, None] * LD + offs_c[None, :], acc, mask=mask)
    tile = s * NTN + j
    old = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
    if old == 2 * SPLIT_K - 1:
        offg = srow[:, None] * LD + offs_g[None, :]
        hg = _red4_store(ws_ptr, NV * LD, offg, mask, SPLIT_K, BLOCK_M, BLOCK_N).to(tl.bfloat16).to(tl.float32)
        hu = _red4_store(ws_ptr, NV * LD, offg + I, mask, SPLIT_K, BLOCK_M, BLOCK_N).to(tl.bfloat16).to(tl.float32)
        sg = libdevice.div_rn(hg, 1.0 + libdevice.exp(-hg))           # = F.silu in fp32 (bitwise, order_probe)
        av = tl.where(valid, sg * hu, 0.0)
        tl.store(act_ptr + srow[:, None] * I + offs_g[None, :], av.to(tl.bfloat16), mask=mask)
        tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit
def _moe2_dec(act_ptr, ids_ptr, tw_ptr, key_ptr, lo_ptr, hi_ptr, out_ptr, ws_ptr, cnt_ptr, NG, H, stride_e_key, stride_e_lo, NTN, NE,
              TOPK: tl.constexpr, NLEV: tl.constexpr, WB: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """grid (NTN, TOPK, T): one token x one expert x one output tile: acc * router weight -> bf16 (= kern.tq_moe's per-slot store) into a
    bf16 workspace; the last of the TOPK programs of a (token, tile) sums the TOPK values in torch's fixed order -> bf16"""
    pid_n = tl.program_id(0); k = tl.program_id(1).to(tl.int64); t = tl.program_id(2).to(tl.int64)
    s = t * TOPK + k
    rr = tl.arange(0, BLOCK_M); m_mask = rr == 0
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < H
    offs_w = tl.arange(0, 32); w_mask = offs_w < 26
    eid = tl.load(ids_ptr + s).to(tl.int64)
    valid = (eid >= 0) & (eid < NE)                                  # MASK (as _moe13_dec)
    e = tl.where(valid, eid, 0)
    am_mask = m_mask & valid
    wt = tl.load(tw_ptr + s).to(tl.float32)
    arow = tl.zeros((BLOCK_M,), dtype=tl.int64) + s
    acc = _tile_acc(act_ptr, arow, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                    offs_n, n_mask, offs_w, w_mask, NG * 128, 0, NG, H, 0, H, 0, NLEV, WB, BLOCK_M, BLOCK_N)
    v = tl.where(valid, (acc * wt).to(tl.bfloat16), 0.0)            # invalid slot -> exactly 0 (even if its router weight is not finite)
    mask = m_mask[:, None] & n_mask[None, :]
    srow = s + rr * 0
    tl.store(ws_ptr + srow[:, None] * H + offs_n[None, :], v.to(tl.bfloat16), mask=mask)
    tile = t * NTN + pid_n
    old = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
    if old == TOPK - 1:
        trow = t + rr * 0
        a0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32); a1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32); a3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in tl.static_range(TOPK):
            q = tl.load(ws_ptr + (trow * TOPK + kk)[:, None] * H + offs_n[None, :], mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            if kk % 4 == 0:
                a0 += q
            elif kk % 4 == 1:
                a1 += q
            elif kk % 4 == 2:
                a2 += q
            else:
                a3 += q
        tl.store(out_ptr + trow[:, None] * H + offs_n[None, :], (((a0 + a1) + a2) + a3).to(tl.bfloat16), mask=mask)
        tl.atomic_xchg(cnt_ptr + tile, 0)


def tq_moe_decode(x, w13, w2, topk_weights, topk_ids):
    """decode-specialised routed experts (T * top_k <= 64), bit-identical to kern.tq_moe on that path (valid ids). Invalid ids
    (< 0 or >= E) -> that slot contributes exactly 0 (masked, = router weight 0)."""
    T, H = x.shape; top_k = topk_ids.shape[1]; NV = T * top_k; E = w13[0].shape[0]
    k13, l13, h13, n13 = w13; k2, l2, h2, n2 = w2
    NG13, I2 = k13.shape[1], k13.shape[2]; I = I2 // 2
    sk = max(1, min(4, NG13 // 4))                                   # = kern.tq_moe's sk1 for T * top_k <= 64
    BM, BN = 16, 64
    ids = topk_ids.reshape(-1).contiguous()
    act = torch.empty(NV, I, device=x.device, dtype=torch.bfloat16)
    ntn = triton.cdiv(I, BN)
    ws = torch.empty(sk, NV, I2, device=x.device, dtype=torch.float32)
    cnt = _counters(("m13", k13.data_ptr()), 64 * ntn)
    hp13 = h13 if h13 is not None else l13
    _moe13_dec[(2 * ntn, NV, sk)](x, ids, k13, l13, hp13, act, ws, cnt, NG13, I, NV, x.stride(0), k13.stride(0), l13.stride(0), ntn, E,
                                  TOPK=top_k, NLEV=n13, WB=16 if n13 == 9 else 24, SPLIT_K=sk, BLOCK_M=BM, BLOCK_N=BN,
                                  num_warps=4, num_stages=2)
    out = torch.empty(T, H, device=x.device, dtype=torch.bfloat16)
    tw = topk_weights.reshape(-1).to(torch.float32)
    hp2 = h2 if h2 is not None else l2
    ntn2 = triton.cdiv(H, BN)
    ws2 = torch.empty(NV, H, device=x.device, dtype=torch.bfloat16)
    cnt2 = _counters(("m2", k2.data_ptr()), 64 * ntn2)
    _moe2_dec[(ntn2, top_k, T)](act, ids, tw, k2, l2, hp2, out, ws2, cnt2, k2.shape[1], H, k2.stride(0), l2.stride(0), ntn2, E,
                                TOPK=top_k, NLEV=n2, WB=16 if n2 == 9 else 24, BLOCK_M=BM, BLOCK_N=BN, num_warps=4, num_stages=2)
    return out


@triton.jit
def _gemv_bf16(x_ptr, w_ptr, out_ptr, M, N, K, stride_xm, stride_wn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN); n_mask = offs_n < N
    offs_m = tl.arange(0, BM); m_mask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK); k_mask = offs_k < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        for m in tl.static_range(BM):
            xr = tl.load(x_ptr + m * stride_xm + offs_k, mask=k_mask & (m < M), other=0.0).to(tl.float32)
            part = tl.sum(w * xr[None, :], axis=1)
            acc += tl.where(offs_m[:, None] == m, part[None, :], 0.0)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def bf16_gemv(x, w):
    """x [M, K] bf16 (M <= 8), w [N, K] bf16 -> [M, N] bf16; fixed order (ORDER-CHANGING vs cuBLAS)."""
    M, K = x.shape; N = w.shape[0]
    BN = 4 if N >= 128 else 1
    BM = 1 if M == 1 else 2 if M == 2 else 4 if M <= 4 else 8
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    _gemv_bf16[(triton.cdiv(N, BN),)](x, w, out, M, N, K, x.stride(0), w.stride(0), BM=BM, BN=BN, BK=512, num_warps=4)
    return out


def flag(name):
    return os.environ.get(name, "0") == "1"

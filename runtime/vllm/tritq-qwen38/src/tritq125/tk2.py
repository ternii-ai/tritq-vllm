# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""tk2.py: v2 Triton kernels for the decode step (dense trit linears with per-shape tiles, batched routed-expert MoE for 64 < slots and
T <= 32 rows, 8-bit hyper-connection GEMM at > 16 rows). Every kernel here is BIT-IDENTICAL to the kernel it replaces (same split-K group
ranges and reduction orders), checked with torch.equal against the reference kernels before any timing.

Levers (each behind its own argument so the microbenchmark can attribute the gain):
  DEC=1   magic-number decode: digit u (0..N-1) -> fp32 via the bit pattern 0x4B000000 | u (= 2^23 + u), minus (2^23 + L) in fp32 (exact),
          then bf16 = the top 16 bits of that fp32 (exact: |q| <= 13 has <= 4 significant bits, so the low 16 bits are zero). Replaces
          int32 -> fp32 -> bf16 (two conversion-pipe instructions, 1/4 rate on the SM) with one OR, one FADD and one shift (full rate).
          Same bf16 values as today -> same tl.dot inputs -> same outputs.
          HC8 (int8): the same trick with u = q + 128.
  BN      free output-tile width for the dense tiles: the per-element arithmetic of tl.dot does not depend on BLOCK_N (the Qwen3.6-35B-A3B plugin uses
          BN 128 bit-identically), so BN can be chosen to balance the grid (qkvz: 256 tiles of 64 = 1.33 waves at 4 CTAs/SM).
  PDL     programmatic dependent launch (Triton >= 3.4: launch_pdl=True, tl.extra.cuda.gdc_wait / gdc_launch_dependents): the kernel may
          start while its predecessor drains; it prefetches its first PF groups of weights into L2 (prefetch.global.L2::evict_last), then
          waits for the predecessor before touching activations. Weights never depend on the predecessor, so this is value-neutral.
  NEXT    after its main loop, each program prefetches its share of the NEXT weight kernel's first bytes into L2 (pointer + byte count
          passed by the caller), so the small kernels in between (norms, top-k, HC mix) overlap with DRAM streaming. Value-neutral.
  OWNER   MoE decode with expert de-duplication ("first-occurrence owner"): of the slots routed to the same expert, the first one's
          program computes ALL of them as rows of one BLOCK_M = 16 tile (decode once, MMA rows independent), the others exit. Same split-K
          group ranges, same fixed-order reductions -> per slot bit-identical to glue.tq_moe_decode (verified per row in the harness).
          Covers M <= 16 rows (T * top_k <= 160): SPLIT_K 4 at T * top_k <= 64 (= today's L1 path), SPLIT_K 1 above (= the numerics of
          today's kern.tq_moe_hc fallback, which X hits at verify width 7 under the confidence stop) -> removes the fallback cliff.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

try:                                                    # PDL primitives (Triton >= 3.4); absent -> PDL arguments are ignored
    from triton.language.extra.cuda import gdc_wait as _gdc_wait, gdc_launch_dependents as _gdc_launch
    HAVE_PDL = True
except Exception:                                       # noqa: BLE001
    HAVE_PDL = False

    @triton.jit
    def _gdc_wait():
        pass

    @triton.jit
    def _gdc_launch():
        pass

from .glue import _counters, _red4_store, _eid  # reference helpers (unchanged)
from .hc8 import _epi

G, NW = 128, 26


# ------------------------------------------------------------------------------------------------ decode helpers
@triton.jit
def _digit_bf16(u, L: tl.constexpr, DEC: tl.constexpr):
    """u int32 in [0, N-1] -> bf16 (u - L), exact, both ways."""
    if DEC == 1:
        f = (u | 0x4B000000).to(tl.float32, bitcast=True) - (8388608.0 + L)
        return (f.to(tl.int32, bitcast=True) >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    else:
        return (u - L).to(tl.bfloat16)


@triton.jit
def _prefetch_l2(ptrs, mask):
    """prefetch.global.L2::evict_last for every (masked) address; returns nothing useful (kept alive by is_pure=False)."""
    tl.inline_asm_elementwise("@$2 prefetch.global.L2::evict_last [$1];\n mov.u32 $0, 0;", "=r,l,b", [ptrs, mask],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _prefetch_span(base_u8, nbytes, pid, nprog, LINE: tl.constexpr, NL: tl.constexpr):
    """this program's 1/nprog share of [base, base + nbytes), one prefetch per 128-B line, NL lines per instruction batch."""
    per = tl.cdiv(tl.cdiv(nbytes, LINE), nprog)
    l0 = pid * per
    for c in range(0, per, NL):
        li = l0 + c + tl.arange(0, NL)
        ok = (li < l0 + per) & (li * LINE < nbytes)
        _prefetch_l2(base_u8 + li.to(tl.int64) * LINE, ok)


@triton.jit
def _tile_acc2(a_ptr, arow, m_mask, key_base, lo_base, hi_base, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1, LD, COL0, LDH, HCOL0,
               NLEV: tl.constexpr, WB: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DEC: tl.constexpr):
    """= kern._tile_acc (same loads, same 5 x tl.dot per group, same fp32 key multiply), with the digit -> bf16 step selectable."""
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    L: tl.constexpr = (NLEV - 1) // 2
    MASKW: tl.constexpr = (1 << WB) - 1
    for g in range(g0, g1):
        wofs = (g * 26 + offs_w[:, None]) * LD + COL0 + offs_n[None, :]
        x = tl.load(lo_base + wofs, mask=w_mask[:, None] & n_mask[None, :], other=0).to(tl.int32) & 0xFFFF
        if WB == 24:
            hofs = (g * 26 + offs_w[:, None]) * LDH + HCOL0 + offs_n[None, :]
            x = x | (tl.load(hi_base + hofs, mask=w_mask[:, None] & n_mask[None, :], other=0).to(tl.int32) << 16)
        kd = tl.load(key_base + g * LD + COL0 + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in tl.static_range(5):
            x = x * NLEV
            q = _digit_bf16(x >> WB, L, DEC)
            x = x & MASKW
            kin = i * 26 + offs_w
            a = tl.load(a_ptr + arow[:, None] * stride_am + g * 128 + kin[None, :],
                        mask=m_mask[:, None] & (w_mask & (kin < 128))[None, :], other=0.0)
            part = tl.dot(a, q, acc=part)
        acc += part * kd[None, :]
    return acc


@triton.jit
def _pf_weights(lo_base, hi_base, offs_n, n_mask, g0, PFG, LD, COL0, LDH, HCOL0, WB: tl.constexpr):
    """prefetch the first PFG groups of this program's lo / hi rows into L2 (before gdc_wait)."""
    offs_r = tl.arange(0, 32)
    for g in range(g0, g0 + PFG):
        rr = g * 26 + offs_r
        m = (offs_r < 26)[:, None] & n_mask[None, :]
        _prefetch_l2((lo_base + (rr[:, None] * LD + COL0 + offs_n[None, :])).to(tl.pointer_type(tl.int8)), m & (offs_n[None, :] % 64 == 0))
        if WB == 24:
            _prefetch_l2((hi_base + (rr[:, None] * LDH + HCOL0 + offs_n[None, :])).to(tl.pointer_type(tl.int8)), m & (offs_n[None, :] % 128 == 0))


# ------------------------------------------------------------------------------------------------ dense (= glue._dense_rk + levers)
@triton.jit
def _dense2(a_ptr, key_ptr, lo_ptr, hi_ptr, out_ptr, ws_ptr, cnt_ptr, M, NOUT, NG, stride_am, LD, LDH, HCOL0, SEGB, NTN,
            nxt_ptr, nxt_bytes,
            NLEV: tl.constexpr, WB: tl.constexpr, NLEV2: tl.constexpr, WB2: tl.constexpr, TWO: tl.constexpr,
            SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DEC: tl.constexpr,
            PDL: tl.constexpr, PFG: tl.constexpr, NEXT: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < NOUT
    offs_w = tl.arange(0, 32); w_mask = offs_w < 26
    arow = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64); m_mask = arow < M
    gper = tl.cdiv(NG, SPLIT_K); g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    if PDL:
        if PFG > 0:
            _pf_weights(lo_ptr, hi_ptr, offs_n, n_mask, g0, tl.minimum(PFG, g1 - g0), LD, 0, LDH, HCOL0, WB)
        _gdc_wait()
    if TWO:
        if pid_n * BLOCK_N >= SEGB:
            acc = _tile_acc2(a_ptr, arow, m_mask, key_ptr, lo_ptr, hi_ptr, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                             LD, 0, LDH, HCOL0, NLEV2, WB2, BLOCK_M, BLOCK_N, DEC)
        else:
            acc = _tile_acc2(a_ptr, arow, m_mask, key_ptr, lo_ptr, hi_ptr, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                             LD, 0, LDH, HCOL0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
    else:
        acc = _tile_acc2(a_ptr, arow, m_mask, key_ptr, lo_ptr, hi_ptr, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                         LD, 0, LDH, HCOL0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
    if PDL:
        _gdc_launch()
    if NEXT:
        npg = tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2)
        lin = (pid_k * tl.num_programs(1) + pid_m) * tl.num_programs(0) + pid_n
        _prefetch_span(nxt_ptr, nxt_bytes, lin, npg, 128, 16)
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


def _cfg_ref(M, NOUT, NG, nprog_target=192):
    """= kern._cfg: the SPLIT_K choice is kept (it defines the group ranges = the numerics); BN is free."""
    from .kern import _cfg
    return _cfg(M, NOUT, NG, nprog_target)


def tq_linear2(x, key, lo, hi, nlev, segs=None, bn=None, warps=4, stages=2, dec=1, pdl=False, pfg=0, nxt=None):
    """= glue.tq_linear_rk (<= 2 segments), with the v2 levers. nxt = (tensor, nbytes) to prefetch into L2 at the end (or None)."""
    M, K = x.shape; NG, LD = key.shape; assert K == NG * G
    if segs is None or len(segs) == 1:
        nl = nlev if segs is None else segs[0][2]
        two, segb, nl2, hshift = False, LD, nl, 0
        if segs is not None and nl == 27: hshift = segs[0][3] - segs[0][0]
    else:
        (c0, n0, nl, h0), (c1, n1, nl2, h1) = segs
        two, segb = True, c1
        hshift = (h0 - c0) if nl == 27 else (h1 - c1)
    BM, BN0, sk = _cfg_ref(M, LD, NG)
    BN = bn or BN0
    if two: assert segb % BN == 0, (segb, BN)
    WB, WB2 = (16 if nl == 9 else 24), (16 if nl2 == 9 else 24)
    hp = hi if hi is not None else lo
    LDH = hi.shape[-1] if hi is not None else LD
    out = torch.empty(M, LD, device=x.device, dtype=torch.bfloat16)
    ntn, ntm = triton.cdiv(LD, BN), triton.cdiv(M, BM)
    if sk > 1:
        ws = torch.empty(sk, M, LD, device=x.device, dtype=torch.float32)
        cnt = _counters(("d2", key.data_ptr(), BN), ntn * 64)
    else:
        ws, cnt = out, out
    nx, nb = (nxt[0], int(nxt[1])) if nxt is not None else (lo, 0)
    kw = dict(launch_pdl=True) if (pdl and HAVE_PDL) else {}
    _dense2[(ntn, ntm, sk)](x, key, lo, hp, out, ws, cnt, M, LD, NG, x.stride(0), LD, LDH, hshift, segb, ntn, nx, nb,
                            NLEV=nl, WB=WB, NLEV2=nl2, WB2=WB2, TWO=two, SPLIT_K=sk, BLOCK_M=BM, BLOCK_N=BN, DEC=dec,
                            PDL=bool(pdl and HAVE_PDL), PFG=pfg, NEXT=nxt is not None, num_warps=warps, num_stages=stages, **kw)
    return out


# ------------------------------------------------------------------------------------------------ MoE decode with expert owners
@triton.jit
def _members(ids_ptr, s, NV, NVP: tl.constexpr, BLOCK_M: tl.constexpr):
    """slots routed to the same id as slot s: (is_owner, member slot index per row [BLOCK_M], row valid [BLOCK_M])."""
    offs = tl.arange(0, NVP)
    idv = tl.load(ids_ptr + offs, mask=offs < NV, other=-2147483647)
    me = tl.load(ids_ptr + s)
    memb = (idv == me) & (offs < NV)
    first = tl.min(tl.where(memb, offs, NVP), axis=0)
    pos = tl.cumsum(memb.to(tl.int32), axis=0) - 1
    rr = tl.arange(0, BLOCK_M)
    hit = (pos[None, :] == rr[:, None]) & memb[None, :]
    sl = tl.sum(tl.where(hit, offs[None, :], 0), axis=1).to(tl.int64)
    cntm = tl.sum(memb.to(tl.int32), axis=0)
    return first == s, sl, rr < cntm


@triton.jit
def _moe13_own(x_ptr, ids_ptr, key_ptr, lo_ptr, hi_ptr, act_ptr, ws_ptr, cnt_ptr, NG, I, NV, stride_xm, stride_e_key, stride_e_lo,
               NTN, NE, emA_ptr, emB_ptr, keyB_ptr, loB_ptr, hiB_ptr,
               TOPK: tl.constexpr, NLEV: tl.constexpr, WB: tl.constexpr, SPLIT_K: tl.constexpr, NVP: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TWO_LEV: tl.constexpr, NLEVB: tl.constexpr, WBB: tl.constexpr,
               DEC: tl.constexpr):
    """= glue._moe13_dec per slot; grid (2 * NTN, NV, SPLIT_K); duplicates exit, the owner computes every member slot as a row."""
    pid_n = tl.program_id(0); s = tl.program_id(1).to(tl.int64); pid_k = tl.program_id(2)
    own, sl, rv = _members(ids_ptr, s, NV, NVP, BLOCK_M)
    if not own:
        return
    half = pid_n // NTN; j = pid_n % NTN
    valid, useA, e = _eid(ids_ptr, s, NE, emA_ptr, emB_ptr, TWO_LEV)
    am_mask = rv & valid
    arow = sl // TOPK
    offs_w = tl.arange(0, 32); w_mask = offs_w < 26
    offs_g = j * BLOCK_N + tl.arange(0, BLOCK_N); g_mask = offs_g < I
    offs_c = offs_g + half * I
    LD = 2 * I
    gper = tl.cdiv(NG, SPLIT_K); g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    if TWO_LEV:
        if useA:
            acc = _tile_acc2(x_ptr, arow, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                             offs_c, g_mask, offs_w, w_mask, stride_xm, g0, g1, LD, 0, LD, 0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
        else:
            acc = _tile_acc2(x_ptr, arow, am_mask, keyB_ptr + e * stride_e_key, loB_ptr + e * stride_e_lo, hiB_ptr + e * stride_e_lo,
                             offs_c, g_mask, offs_w, w_mask, stride_xm, g0, g1, LD, 0, LD, 0, NLEVB, WBB, BLOCK_M, BLOCK_N, DEC)
    else:
        acc = _tile_acc2(x_ptr, arow, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                         offs_c, g_mask, offs_w, w_mask, stride_xm, g0, g1, LD, 0, LD, 0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
    mask = rv[:, None] & g_mask[None, :]
    # SPLIT_K 1 (verify widths > 6 rows): the single partial goes through the same workspace and the same bf16 rounding after the
    # (trivial) reduction = kern.tq_moe(_hc)'s direct bf16 store of w13, then the torch SiLU * up glue (bitwise, glue.order_probe).
    tl.store(ws_ptr + pid_k * NV * LD + sl[:, None] * LD + offs_c[None, :], acc, mask=mask)
    target = 2 * SPLIT_K - 1
    tilev = sl * NTN + j
    old = tl.atomic_add(cnt_ptr + tilev, 1, mask=rv, sem="acq_rel")
    last = rv & (old == target)
    lmask = last[:, None] & g_mask[None, :]
    offg = sl[:, None] * LD + offs_g[None, :]
    hg = _red4_store(ws_ptr, NV * LD, offg, lmask, SPLIT_K, BLOCK_M, BLOCK_N).to(tl.bfloat16).to(tl.float32)
    hu = _red4_store(ws_ptr, NV * LD, offg + I, lmask, SPLIT_K, BLOCK_M, BLOCK_N).to(tl.bfloat16).to(tl.float32)
    sg = libdevice.div_rn(hg, 1.0 + libdevice.exp(-hg))
    av = tl.where(valid, sg * hu, 0.0)
    tl.store(act_ptr + sl[:, None] * I + offs_g[None, :], av.to(tl.bfloat16), mask=lmask)
    tl.atomic_xchg(cnt_ptr + tilev, 0, mask=last)


@triton.jit
def _moe2_own(act_ptr, ids_ptr, tw_ptr, key_ptr, lo_ptr, hi_ptr, out_ptr, ws_ptr, cnt_ptr, NG, H, NV, stride_e_key, stride_e_lo, NTN, NE,
              emA_ptr, emB_ptr, keyB_ptr, loB_ptr, hiB_ptr,
              TOPK: tl.constexpr, NLEV: tl.constexpr, WB: tl.constexpr, NVP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              TWO_LEV: tl.constexpr, NLEVB: tl.constexpr, WBB: tl.constexpr, DEC: tl.constexpr):
    """= glue._moe2_dec per slot; grid (NTN, TOPK, T); the owner computes every member slot; per-token top-k sum by the last arriver."""
    pid_n = tl.program_id(0); k = tl.program_id(1).to(tl.int64); t = tl.program_id(2).to(tl.int64)
    s = t * TOPK + k
    own, sl, rv = _members(ids_ptr, s, NV, NVP, BLOCK_M)
    if not own:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < H
    offs_w = tl.arange(0, 32); w_mask = offs_w < 26
    valid, useA, e = _eid(ids_ptr, s, NE, emA_ptr, emB_ptr, TWO_LEV)
    am_mask = rv & valid
    wt = tl.load(tw_ptr + sl, mask=rv, other=0.0).to(tl.float32)
    if TWO_LEV:
        if useA:
            acc = _tile_acc2(act_ptr, sl, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                             offs_n, n_mask, offs_w, w_mask, NG * 128, 0, NG, H, 0, H, 0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
        else:
            acc = _tile_acc2(act_ptr, sl, am_mask, keyB_ptr + e * stride_e_key, loB_ptr + e * stride_e_lo, hiB_ptr + e * stride_e_lo,
                             offs_n, n_mask, offs_w, w_mask, NG * 128, 0, NG, H, 0, H, 0, NLEVB, WBB, BLOCK_M, BLOCK_N, DEC)
    else:
        acc = _tile_acc2(act_ptr, sl, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                         offs_n, n_mask, offs_w, w_mask, NG * 128, 0, NG, H, 0, H, 0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
    v = tl.where(valid, (acc * wt[:, None]).to(tl.bfloat16), 0.0)
    mask = rv[:, None] & n_mask[None, :]
    tl.store(ws_ptr + sl[:, None] * H + offs_n[None, :], v.to(tl.bfloat16), mask=mask)
    tok = sl // TOPK
    tilev = tok * NTN + pid_n
    old = tl.atomic_add(cnt_ptr + tilev, 1, mask=rv, sem="acq_rel")
    last = rv & (old == TOPK - 1)
    lmask = last[:, None] & n_mask[None, :]
    a0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32); a1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32); a3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in tl.static_range(TOPK):
        q = tl.load(ws_ptr + (tok * TOPK + kk)[:, None] * H + offs_n[None, :], mask=lmask, other=0.0, cache_modifier=".cg").to(tl.float32)
        if kk % 4 == 0:
            a0 += q
        elif kk % 4 == 1:
            a1 += q
        elif kk % 4 == 2:
            a2 += q
        else:
            a3 += q
    tl.store(out_ptr + tok[:, None] * H + offs_n[None, :], (((a0 + a1) + a2) + a3).to(tl.bfloat16), mask=lmask)
    tl.atomic_xchg(cnt_ptr + tilev, 0, mask=last)


def tq_moe_decode2(x, w13, w2, topk_weights, topk_ids, dec=1, sk=None):
    """= glue.tq_moe_decode at T * top_k <= 64 (SPLIT_K 4) and = the numerics of kern.tq_moe(_hc) above (SPLIT_K 1), for T * top_k <= 160
    (verify widths up to 16 rows), with the expert-owner de-duplication. Same argument conventions as glue.tq_moe_decode."""
    from .glue import _split_groups
    g13 = w13 if isinstance(w13, list) else [w13]
    g2 = w2 if isinstance(w2, list) else [w2]
    A13, B13, emA13, emB13 = _split_groups(g13)
    A2, B2, emA2, emB2 = _split_groups(g2)
    two = B13 is not None
    Tx, H = x.shape; T = min(Tx, topk_ids.shape[0]); top_k = topk_ids.shape[1]; NV = T * top_k
    assert NV <= 160, NV
    NVP = triton.next_power_of_2(NV)
    E = emA13.shape[0] if emA13 is not None else A13[0].shape[0]
    k13, l13, h13, n13 = A13[:4]; k2, l2, h2, n2 = A2[:4]
    NG13, I2 = k13.shape[1], k13.shape[2]; I = I2 // 2
    if sk is None:
        sk = max(1, min(4, NG13 // 4)) if NV <= 64 else 1
    BM, BN = 16, 64
    ids = topk_ids[:T].reshape(-1).contiguous()
    act = torch.empty(NV, I, device=x.device, dtype=torch.bfloat16)
    ntn = triton.cdiv(I, BN)
    ws = torch.empty(max(sk, 1), NV, I2, device=x.device, dtype=torch.float32)
    cnt = _counters(("m13o", k13.data_ptr()), 160 * ntn)
    hp13 = h13 if h13 is not None else l13
    if two:
        kb, lb, hb, nb = B13[:4]; hpb = hb if hb is not None else lb
        ex13 = dict(emA_ptr=emA13, emB_ptr=emB13, keyB_ptr=kb, loB_ptr=lb, hiB_ptr=hpb)
        kw13 = dict(TWO_LEV=True, NLEVB=nb, WBB=16 if nb == 9 else 24)
    else:
        ex13 = dict(emA_ptr=ids, emB_ptr=ids, keyB_ptr=k13, loB_ptr=l13, hiB_ptr=hp13)
        kw13 = dict(TWO_LEV=False, NLEVB=9, WBB=16)
    _moe13_own[(2 * ntn, NV, sk)](x, ids, k13, l13, hp13, act, ws, cnt, NG13, I, NV, x.stride(0), k13.stride(0), l13.stride(0), ntn, E,
                                  **ex13, TOPK=top_k, NLEV=n13, WB=16 if n13 == 9 else 24, SPLIT_K=sk, NVP=NVP, BLOCK_M=BM, BLOCK_N=BN,
                                  DEC=dec, num_warps=4, num_stages=2, **kw13)
    out = torch.empty(Tx, H, device=x.device, dtype=torch.bfloat16)
    tw = topk_weights[:T].reshape(-1).to(torch.float32)
    hp2 = h2 if h2 is not None else l2
    ntn2 = triton.cdiv(H, BN)
    ws2 = torch.empty(NV, H, device=x.device, dtype=torch.bfloat16)
    cnt2 = _counters(("m2o", k2.data_ptr()), 16 * ntn2)
    if two:
        kb, lb, hb, nb = B2[:4]; hpb = hb if hb is not None else lb
        ex2 = dict(emA_ptr=emA2, emB_ptr=emB2, keyB_ptr=kb, loB_ptr=lb, hiB_ptr=hpb)
        kw2 = dict(TWO_LEV=True, NLEVB=nb, WBB=16 if nb == 9 else 24)
    else:
        ex2 = dict(emA_ptr=ids, emB_ptr=ids, keyB_ptr=k2, loB_ptr=l2, hiB_ptr=hp2)
        kw2 = dict(TWO_LEV=False, NLEVB=9, WBB=16)
    _moe2_own[(ntn2, top_k, T)](act, ids, tw, k2, l2, hp2, out, ws2, cnt2, k2.shape[1], H, NV, k2.stride(0), l2.stride(0), ntn2, E,
                                **ex2, TOPK=top_k, NLEV=n2, WB=16 if n2 == 9 else 24, NVP=NVP, BLOCK_M=BM, BLOCK_N=BN, DEC=dec,
                                num_warps=4, num_stages=2, **kw2)
    return out


# ------------------------------------------------------------------------------------------------ HC8 int8 (= hc8._q8_kernel + levers)
@triton.jit
def _i8_bf16(w, DEC: tl.constexpr):
    if DEC == 1:
        u = w.to(tl.int32) + 128
        f = (u | 0x4B000000).to(tl.float32, bitcast=True) - 8388736.0          # 2^23 + 128
        return (f.to(tl.int32, bitcast=True) >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    else:
        return w.to(tl.bfloat16)


@triton.jit
def _q8_kernel2(x_ptr, q_ptr, s_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, NG, stride_xm, NTN, nxt_ptr, nxt_bytes,
                SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr,
                SILU_N: tl.constexpr, HC: tl.constexpr, DEC: tl.constexpr, PDL: tl.constexpr, NEXT: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < N
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64); m_mask = offs_m < M
    offs_g = tl.arange(0, 32)
    gper = tl.cdiv(NG, SPLIT_K); g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    if PDL:
        kk0 = g0 * 32 + offs_g
        _prefetch_l2((q_ptr + kk0[:, None].to(tl.int64) * N + offs_n[None, :]).to(tl.pointer_type(tl.int8)),
                     n_mask[None, :] & (offs_n[None, :] % 128 == 0))
        _gdc_wait()
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(g0, g1, GPI):
        for u in tl.static_range(GPI):
            gg = g + u
            ok = gg < g1
            kk = gg * 32 + offs_g
            a = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None] & ok, other=0.0)
            w = _i8_bf16(tl.load(q_ptr + kk[:, None].to(tl.int64) * N + offs_n[None, :], mask=n_mask[None, :] & ok, other=0), DEC)
            sc = tl.load(s_ptr + gg * N + offs_n, mask=n_mask & ok, other=0.0).to(tl.float32)
            part = tl.dot(a, w)
            acc += part * sc[None, :]
    if PDL:
        _gdc_launch()
    if NEXT:
        npg = tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2)
        lin = (pid_k * tl.num_programs(1) + pid_m) * tl.num_programs(0) + pid_n
        _prefetch_span(nxt_ptr, nxt_bytes, lin, npg, 128, 16)
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


def q8_linear2(x, q, s, silu_n=0, hc=1, dec=1, pdl=False, nxt=None, bn=None, split=None, warps=None, stages=None, gpi=None, bm=None):
    """= hc8.q8_linear (same TUNED table / split -> same numerics), with the v2 levers."""
    from . import hc8
    M, K = x.shape; N = q.shape[1]; NG = K // 32
    t = hc8.TUNED.get((N, K)) if M <= 16 else None
    if t is not None and bn is None and split is None:
        bn, split, warps, stages, gpi = t
    BM, BN, sk = hc8._cfg(M, N, NG, bn)
    if split is not None: sk = split
    if bm is not None: BM = bm                     # rows per tile: value-neutral (MMA rows independent); split kept = numerics kept
    warps, stages, gpi = warps or 4, stages or 3, gpi or 1
    ntn, ntm = triton.cdiv(N, BN), triton.cdiv(M, BM)
    out = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if sk > 1:
        ws = torch.empty(sk, M, N, device=x.device, dtype=torch.float32)
        cnt = _counters(("hc8v2", q.data_ptr(), BN), ntn * 64)
    else:
        ws, cnt = out, out
    nx, nb = (nxt[0], int(nxt[1])) if nxt is not None else (q, 0)
    kw = dict(launch_pdl=True) if (pdl and HAVE_PDL) else {}
    _q8_kernel2[(ntn, ntm, sk)](x, q, s, out, ws, cnt, M, N, NG, x.stride(0), ntn, nx, nb, SPLIT_K=sk, BLOCK_M=BM, BLOCK_N=BN, GPI=gpi,
                                SILU_N=silu_n, HC=hc, DEC=dec, PDL=bool(pdl and HAVE_PDL), NEXT=nxt is not None,
                                num_warps=warps, num_stages=stages, **kw)
    return out


# ------------------------------------------------------------------------------------------------ batched MoE (verify at C users: slots <= 1024)
# A one-program PLAN kernel sorts the slots by (expert id, slot) on the device (tl.sort, deterministic) and writes segments: perm[NV] =
# slots in that order, seg[d] = start of distinct expert d, ndist. Invalid ids (< 0 or >= E) get singleton segments (output exactly 0).
# The GEMM grid is (tiles, DMAX = NV, chunks x SPLIT_K): program (d, chunk) computes rows perm[seg[d] + 16*chunk ...] (<= 16 rows) of expert d;
# programs beyond ndist or beyond the segment exit. Per row the arithmetic is the BLOCK_M 16 tile of _tile_acc2 over the same group range,
# so per slot it equals the fallback kern.tq_moe(_hc) (SPLIT_K 1, > 64 slots) and the L1 decode (SPLIT_K 4, <= 64 slots). Decode is done
# once per (expert, 16-row chunk): at C = 8 the mean is 2.2 rows per distinct expert in a measured routed batch, so nearly always once.
@triton.jit
def _moe_plan(ids_ptr, perm_ptr, seg_ptr, nd_ptr, key_ws, NV, NE, NVP: tl.constexpr):
    offs = tl.arange(0, NVP); ok = offs < NV
    e = tl.load(ids_ptr + offs, mask=ok, other=0).to(tl.int64)
    valid = (e >= 0) & (e < NE)
    eg = tl.where(valid, e, NE + offs)                                  # invalid slot -> its own pseudo-expert
    key = tl.where(ok, eg * 2048 + offs, 0x7FFFFFFFFFFF)
    ks = tl.sort(key)
    tl.store(key_ws + offs, ks)
    tl.debug_barrier()
    prev = tl.load(key_ws + offs - 1, mask=offs > 0, other=-1)
    eid_s = ks // 2048; slot_s = ks % 2048
    okk = offs < NV
    first = okk & ((offs == 0) | (eid_s != prev // 2048))
    segid = tl.cumsum(first.to(tl.int32), axis=0) - 1
    tl.store(perm_ptr + offs, slot_s.to(tl.int32), mask=okk)
    tl.store(seg_ptr + segid, offs.to(tl.int32), mask=first)
    nd = tl.sum(first.to(tl.int32), axis=0)
    tl.store(seg_ptr + nd, NV)                                          # sentinel end
    tl.store(nd_ptr, nd)


@triton.jit
def _seg_rows(perm_ptr, seg_ptr, nd_ptr, d, chunk, BLOCK_M: tl.constexpr):
    nd = tl.load(nd_ptr)
    s0 = tl.load(seg_ptr + d, mask=d < nd, other=0); s1 = tl.load(seg_ptr + d + 1, mask=d < nd, other=0)
    rr = tl.arange(0, BLOCK_M); pos = s0 + chunk * BLOCK_M + rr
    rv = (d < nd) & (pos < s1)
    sl = tl.load(perm_ptr + pos, mask=rv, other=0).to(tl.int64)
    return sl, rv, (d < nd) & (s0 + chunk * BLOCK_M < s1)


@triton.jit
def _moe13_seg(x_ptr, ids_ptr, perm_ptr, seg_ptr, nd_ptr, key_ptr, lo_ptr, hi_ptr, act_ptr, ws_ptr, cnt_ptr, NG, I, NV, stride_xm,
               stride_e_key, stride_e_lo, NTN, NE, emA_ptr, emB_ptr, keyB_ptr, loB_ptr, hiB_ptr,
               TOPK: tl.constexpr, NLEV: tl.constexpr, WB: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, TWO_LEV: tl.constexpr, NLEVB: tl.constexpr, WBB: tl.constexpr, DEC: tl.constexpr):
    pid_n = tl.program_id(0); d = tl.program_id(1); pz = tl.program_id(2)
    chunk = pz // SPLIT_K; pid_k = pz % SPLIT_K
    sl, rv, live = _seg_rows(perm_ptr, seg_ptr, nd_ptr, d, chunk, BLOCK_M)
    if not live:
        return
    s = tl.sum(tl.where(tl.arange(0, BLOCK_M) == 0, sl, 0), axis=0)    # first row's slot -> expert
    half = pid_n // NTN; j = pid_n % NTN
    valid, useA, e = _eid(ids_ptr, s, NE, emA_ptr, emB_ptr, TWO_LEV)
    am_mask = rv & valid
    arow = sl // TOPK
    offs_w = tl.arange(0, 32); w_mask = offs_w < 26
    offs_g = j * BLOCK_N + tl.arange(0, BLOCK_N); g_mask = offs_g < I
    offs_c = offs_g + half * I
    LD = 2 * I
    gper = tl.cdiv(NG, SPLIT_K); g0 = pid_k * gper; g1 = tl.minimum(g0 + gper, NG)
    if TWO_LEV:
        if useA:
            acc = _tile_acc2(x_ptr, arow, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                             offs_c, g_mask, offs_w, w_mask, stride_xm, g0, g1, LD, 0, LD, 0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
        else:
            acc = _tile_acc2(x_ptr, arow, am_mask, keyB_ptr + e * stride_e_key, loB_ptr + e * stride_e_lo, hiB_ptr + e * stride_e_lo,
                             offs_c, g_mask, offs_w, w_mask, stride_xm, g0, g1, LD, 0, LD, 0, NLEVB, WBB, BLOCK_M, BLOCK_N, DEC)
    else:
        acc = _tile_acc2(x_ptr, arow, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                         offs_c, g_mask, offs_w, w_mask, stride_xm, g0, g1, LD, 0, LD, 0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
    mask = rv[:, None] & g_mask[None, :]
    tl.store(ws_ptr + pid_k * NV * LD + sl[:, None] * LD + offs_c[None, :], acc, mask=mask)
    tilev = sl * NTN + j
    old = tl.atomic_add(cnt_ptr + tilev, 1, mask=rv, sem="acq_rel")
    last = rv & (old == 2 * SPLIT_K - 1)
    lmask = last[:, None] & g_mask[None, :]
    offg = sl[:, None] * LD + offs_g[None, :]
    hg = _red4_store(ws_ptr, NV * LD, offg, lmask, SPLIT_K, BLOCK_M, BLOCK_N).to(tl.bfloat16).to(tl.float32)
    hu = _red4_store(ws_ptr, NV * LD, offg + I, lmask, SPLIT_K, BLOCK_M, BLOCK_N).to(tl.bfloat16).to(tl.float32)
    sg = libdevice.div_rn(hg, 1.0 + libdevice.exp(-hg))
    av = tl.where(valid, sg * hu, 0.0)
    tl.store(act_ptr + sl[:, None] * I + offs_g[None, :], av.to(tl.bfloat16), mask=lmask)
    tl.atomic_xchg(cnt_ptr + tilev, 0, mask=last)


@triton.jit
def _moe2_seg(act_ptr, ids_ptr, perm_ptr, seg_ptr, nd_ptr, tw_ptr, key_ptr, lo_ptr, hi_ptr, out_ptr, ws_ptr, cnt_ptr, NG, H, NV,
              stride_e_key, stride_e_lo, NTN, NE, emA_ptr, emB_ptr, keyB_ptr, loB_ptr, hiB_ptr,
              TOPK: tl.constexpr, NLEV: tl.constexpr, WB: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              TWO_LEV: tl.constexpr, NLEVB: tl.constexpr, WBB: tl.constexpr, DEC: tl.constexpr):
    pid_n = tl.program_id(0); d = tl.program_id(1); chunk = tl.program_id(2)
    sl, rv, live = _seg_rows(perm_ptr, seg_ptr, nd_ptr, d, chunk, BLOCK_M)
    if not live:
        return
    s = tl.sum(tl.where(tl.arange(0, BLOCK_M) == 0, sl, 0), axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); n_mask = offs_n < H
    offs_w = tl.arange(0, 32); w_mask = offs_w < 26
    valid, useA, e = _eid(ids_ptr, s, NE, emA_ptr, emB_ptr, TWO_LEV)
    am_mask = rv & valid
    wt = tl.load(tw_ptr + sl, mask=rv, other=0.0).to(tl.float32)
    if TWO_LEV:
        if useA:
            acc = _tile_acc2(act_ptr, sl, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                             offs_n, n_mask, offs_w, w_mask, NG * 128, 0, NG, H, 0, H, 0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
        else:
            acc = _tile_acc2(act_ptr, sl, am_mask, keyB_ptr + e * stride_e_key, loB_ptr + e * stride_e_lo, hiB_ptr + e * stride_e_lo,
                             offs_n, n_mask, offs_w, w_mask, NG * 128, 0, NG, H, 0, H, 0, NLEVB, WBB, BLOCK_M, BLOCK_N, DEC)
    else:
        acc = _tile_acc2(act_ptr, sl, am_mask, key_ptr + e * stride_e_key, lo_ptr + e * stride_e_lo, hi_ptr + e * stride_e_lo,
                         offs_n, n_mask, offs_w, w_mask, NG * 128, 0, NG, H, 0, H, 0, NLEV, WB, BLOCK_M, BLOCK_N, DEC)
    v = tl.where(valid, (acc * wt[:, None]).to(tl.bfloat16), 0.0)
    mask = rv[:, None] & n_mask[None, :]
    tl.store(ws_ptr + sl[:, None] * H + offs_n[None, :], v.to(tl.bfloat16), mask=mask)
    tok = sl // TOPK
    tilev = tok * NTN + pid_n
    old = tl.atomic_add(cnt_ptr + tilev, 1, mask=rv, sem="acq_rel")
    last = rv & (old == TOPK - 1)
    lmask = last[:, None] & n_mask[None, :]
    a0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32); a1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32); a3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in tl.static_range(TOPK):
        q = tl.load(ws_ptr + (tok * TOPK + kk)[:, None] * H + offs_n[None, :], mask=lmask, other=0.0, cache_modifier=".cg").to(tl.float32)
        if kk % 4 == 0:
            a0 += q
        elif kk % 4 == 1:
            a1 += q
        elif kk % 4 == 2:
            a2 += q
        else:
            a3 += q
    tl.store(out_ptr + tok[:, None] * H + offs_n[None, :], (((a0 + a1) + a2) + a3).to(tl.bfloat16), mask=lmask)
    tl.atomic_xchg(cnt_ptr + tilev, 0, mask=last)


def tq_moe_batched(x, w13, w2, topk_weights, topk_ids, dec=1, sk=None):
    """routed experts for T tokens (verify rows of C users), T * top_k <= 1024, T <= 64: per slot = glue.tq_moe_decode (<= 64 slots, SPLIT_K 4)
    or kern.tq_moe(_hc) (> 64 slots, SPLIT_K 1). Static shapes, no host sync (graph-capturable)."""
    from .glue import _split_groups
    g13 = w13 if isinstance(w13, list) else [w13]
    g2 = w2 if isinstance(w2, list) else [w2]
    A13, B13, emA13, emB13 = _split_groups(g13)
    A2, B2, emA2, emB2 = _split_groups(g2)
    two = B13 is not None
    Tx, H = x.shape; T = min(Tx, topk_ids.shape[0]); top_k = topk_ids.shape[1]; NV = T * top_k
    assert NV <= 1024 and T <= 64, (T, NV)
    NVP = max(16, triton.next_power_of_2(NV))
    E = emA13.shape[0] if emA13 is not None else A13[0].shape[0]
    k13, l13, h13, n13 = A13[:4]; k2, l2, h2, n2 = A2[:4]
    NG13, I2 = k13.shape[1], k13.shape[2]; I = I2 // 2
    if sk is None:
        sk = max(1, min(4, NG13 // 4)) if NV <= 64 else 1
    BM, BN = 16, 64
    NCH = triton.cdiv(T, BM)                                       # max rows of one expert = T (top-k ids of a token are distinct)
    ids = topk_ids[:T].reshape(-1).contiguous()
    dev = x.device
    perm = torch.empty(NVP, dtype=torch.int32, device=dev); seg = torch.empty(NVP + 1, dtype=torch.int32, device=dev)
    nd = torch.empty(1, dtype=torch.int32, device=dev); kws = torch.empty(NVP, dtype=torch.int64, device=dev)
    _moe_plan[(1,)](ids, perm, seg, nd, kws, NV, E, NVP=NVP, num_warps=4)
    act = torch.empty(NV, I, device=dev, dtype=torch.bfloat16)
    ntn = triton.cdiv(I, BN)
    ws = torch.empty(sk, NV, I2, device=dev, dtype=torch.float32)
    cnt = _counters(("m13b", k13.data_ptr()), 1024 * ntn)
    hp13 = h13 if h13 is not None else l13
    if two:
        kb, lb, hb, nb = B13[:4]; hpb = hb if hb is not None else lb
        ex13 = dict(emA_ptr=emA13, emB_ptr=emB13, keyB_ptr=kb, loB_ptr=lb, hiB_ptr=hpb)
        kw13 = dict(TWO_LEV=True, NLEVB=nb, WBB=16 if nb == 9 else 24)
    else:
        ex13 = dict(emA_ptr=ids, emB_ptr=ids, keyB_ptr=k13, loB_ptr=l13, hiB_ptr=hp13)
        kw13 = dict(TWO_LEV=False, NLEVB=9, WBB=16)
    _moe13_seg[(2 * ntn, NV, NCH * sk)](x, ids, perm, seg, nd, k13, l13, hp13, act, ws, cnt, NG13, I, NV, x.stride(0), k13.stride(0),
                                        l13.stride(0), ntn, E, **ex13, TOPK=top_k, NLEV=n13, WB=16 if n13 == 9 else 24, SPLIT_K=sk,
                                        BLOCK_M=BM, BLOCK_N=BN, DEC=dec, num_warps=4, num_stages=2, **kw13)
    out = torch.empty(Tx, H, device=dev, dtype=torch.bfloat16)
    tw = topk_weights[:T].reshape(-1).to(torch.float32)
    hp2 = h2 if h2 is not None else l2
    ntn2 = triton.cdiv(H, BN)
    ws2 = torch.empty(NV, H, device=dev, dtype=torch.bfloat16)
    cnt2 = _counters(("m2b", k2.data_ptr()), 64 * ntn2)
    if two:
        kb, lb, hb, nb = B2[:4]; hpb = hb if hb is not None else lb
        ex2 = dict(emA_ptr=emA2, emB_ptr=emB2, keyB_ptr=kb, loB_ptr=lb, hiB_ptr=hpb)
        kw2 = dict(TWO_LEV=True, NLEVB=nb, WBB=16 if nb == 9 else 24)
    else:
        ex2 = dict(emA_ptr=ids, emB_ptr=ids, keyB_ptr=k2, loB_ptr=l2, hiB_ptr=hp2)
        kw2 = dict(TWO_LEV=False, NLEVB=9, WBB=16)
    _moe2_seg[(ntn2, NV, NCH)](act, ids, perm, seg, nd, tw, k2, l2, hp2, out, ws2, cnt2, k2.shape[1], H, NV, k2.stride(0), l2.stride(0),
                               ntn2, E, **ex2, TOPK=top_k, NLEV=n2, WB=16 if n2 == 9 else 24, BLOCK_M=BM, BLOCK_N=BN, DEC=dec,
                               num_warps=4, num_stages=2, **kw2)
    return out

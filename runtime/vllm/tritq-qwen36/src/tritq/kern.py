# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""Triton kernels for dense-trit weights (TQ9_128 / TQ27_128, layout in fmt.py), bf16 activations, fp32 accumulation.
One kernel serves dense linears and MoE experts (MOE=True: rows gathered through vLLM's moe_align_block_size sort, B offset per expert,
optional router-weight multiply). Per 128-group: load the 26 words for BLOCK_N outputs once, peel the 5 digits by multiply/shift
(digit i covers inputs i*26 .. i*26+25 of the group), tl.dot each digit chunk (integer codes are exact in bf16) and scale the group's
partial sum by its fp16 key in fp32. SPLIT_K splits the groups across programs (partials summed afterwards, deterministic)."""
import torch
import triton
import triton.language as tl

G, NW = 128, 26


@triton.jit
def _tile_acc(a_ptr, arow, m_mask, key_base, lo_base, hi_base, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1, LD, COL0, LDH, HCOL0,
              NLEV: tl.constexpr, WB: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
            q = ((x >> WB) - L).to(tl.bfloat16)
            x = x & MASKW
            kin = i * 26 + offs_w
            a = tl.load(a_ptr + arow[:, None] * stride_am + g * 128 + kin[None, :],
                        mask=m_mask[:, None] & (w_mask & (kin < 128))[None, :], other=0.0)
            part = tl.dot(a, q, acc=part)
        acc += part * kd[None, :]
    return acc


@triton.jit
def _tq_kernel(a_ptr, key_ptr, lo_ptr, hi_ptr, c_ptr, tw_ptr, sorted_ptr, eid_ptr, ntp_ptr,
               M, NOUT, NG, stride_am, stride_cs, stride_cm, stride_e_key, stride_e_lo, num_valid, top_k,
               LD, COL0, LDH, HCOL0, SEGB,
               NLEV: tl.constexpr, WB: tl.constexpr, NLEV2: tl.constexpr, WB2: tl.constexpr, TWO: tl.constexpr,
               MOE: tl.constexpr, MUL_W: tl.constexpr, A_PER_SLOT: tl.constexpr,
               SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """TWO: columns >= SEGB (a multiple of BLOCK_N) use NLEV2 / WB2 (per-part levels of a fused module, one launch);
    hi column index = column + HCOL0 for the TQ27 tiles."""
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < NOUT
    offs_w = tl.arange(0, 32)
    w_mask = offs_w < 26
    if MOE:
        ntp = tl.load(ntp_ptr)
        if pid_m * BLOCK_M >= ntp:
            return
        offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        slot = tl.load(sorted_ptr + offs_tid).to(tl.int64)
        m_mask = slot < num_valid
        if A_PER_SLOT:
            arow = slot
        else:
            arow = slot // top_k
        crow = slot
        e = tl.load(eid_ptr + pid_m).to(tl.int64)
        key_base = key_ptr + e * stride_e_key
        lo_base = lo_ptr + e * stride_e_lo
        hi_base = hi_ptr + e * stride_e_lo
    else:
        arow = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = arow < M
        crow = arow
        key_base = key_ptr
        lo_base = lo_ptr
        hi_base = hi_ptr
    gper = tl.cdiv(NG, SPLIT_K)
    g0 = pid_k * gper
    g1 = tl.minimum(g0 + gper, NG)
    if TWO:
        if pid_n * BLOCK_N >= SEGB:
            acc = _tile_acc(a_ptr, arow, m_mask, key_base, lo_base, hi_base, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                            LD, COL0, LDH, HCOL0, NLEV2, WB2, BLOCK_M, BLOCK_N)
        else:
            acc = _tile_acc(a_ptr, arow, m_mask, key_base, lo_base, hi_base, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                            LD, COL0, LDH, HCOL0, NLEV, WB, BLOCK_M, BLOCK_N)
    else:
        acc = _tile_acc(a_ptr, arow, m_mask, key_base, lo_base, hi_base, offs_n, n_mask, offs_w, w_mask, stride_am, g0, g1,
                        LD, COL0, LDH, HCOL0, NLEV, WB, BLOCK_M, BLOCK_N)
    if MOE:
        if MUL_W:
            wt = tl.load(tw_ptr + slot, mask=m_mask, other=0.0).to(tl.float32)
            acc = acc * wt[:, None]
    cp = c_ptr + pid_k * stride_cs + crow[:, None] * stride_cm + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(cp, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])
    else:
        tl.store(cp, acc, mask=m_mask[:, None] & n_mask[None, :])


def _cfg(M, NOUT, NG, nprog_target=192):
    BM = 16 if M <= 16 else 32 if M <= 32 else 64
    if NG <= 4:                     # K <= 512 (shared_down): BLOCK_N 32, no split; bit-identical to the split = NG it replaces
        return BM, 32, 1            # (tuned: 7.1 -> 5.3 us at M=1, rel-err 0)
    BN = 64
    tiles = triton.cdiv(NOUT, BN) * triton.cdiv(M, BM)
    split = 1
    while tiles * split < nprog_target and split * 2 <= NG and NG % (split * 2) == 0:
        split *= 2
    return BM, BN, split


def tq_linear(x, key, lo, hi, nlev, out_dtype=torch.bfloat16, split=None, segs=None, bn=None, warps=None):
    """x [M, K] bf16; key [G, N], lo [G, 26, N], hi [G, 26, N27] -> [M, N].
    segs: None (one container nlev over all N columns, hi has N columns) or (col0, ncol, nlev, hcol0) column segments of a fused
    module with per-part levels (hi holds only the TQ27 segments' columns, hcol0 = their offset in hi). Two segments run in ONE
    launch (level chosen per BLOCK_N tile); more segments fall back to one launch each."""
    M, K = x.shape; NG, LD = key.shape; assert K == NG * G
    if segs is None or len(segs) == 1:
        nl = nlev if segs is None else segs[0][2]
        two, segb, nl2, hshift = False, LD, nl, 0
        if segs is not None and nl == 27:
            hshift = segs[0][3] - segs[0][0]
    elif len(segs) == 2:
        (c0, n0, nl, h0), (c1, n1, nl2, h1) = segs
        assert c0 == 0 and c1 == n0 and n0 % 64 == 0, segs
        two, segb = True, c1
        hshift = (h0 - c0) if nl == 27 else (h1 - c1)
    else:
        out = torch.empty(M, LD, device=x.device, dtype=out_dtype)
        for s_ in segs:
            out[:, s_[0]: s_[0] + s_[1]] = tq_linear(x, key[:, s_[0]: s_[0] + s_[1]].contiguous(), lo[:, :, s_[0]: s_[0] + s_[1]].contiguous(),
                                                     hi[:, :, s_[3]: s_[3] + s_[1]].contiguous() if s_[2] == 27 else None, s_[2], out_dtype, split)
        return out
    BM, BN, sk = _cfg(M, LD, NG)
    if split is not None: sk = split
    if bn is not None: BN = bn
    WB, WB2 = (16 if nl == 9 else 24), (16 if nl2 == 9 else 24)
    hp = hi if hi is not None else lo
    LDH = hi.shape[-1] if hi is not None else LD
    if sk == 1:
        out = torch.empty(M, LD, device=x.device, dtype=out_dtype); cs = 0
    else:
        out = torch.empty(sk, M, LD, device=x.device, dtype=torch.float32); cs = M * LD
    grid = (triton.cdiv(LD, BN), triton.cdiv(M, BM), sk)
    _tq_kernel[grid](x, key, lo, hp, out, x, x, x, x, M, LD, NG, x.stride(0), cs, LD, 0, 0, 0, 1,
                     LD, 0, LDH, hshift, segb,
                     NLEV=nl, WB=WB, NLEV2=nl2, WB2=WB2, TWO=two, MOE=False, MUL_W=False, A_PER_SLOT=False, SPLIT_K=sk,
                     BLOCK_M=BM, BLOCK_N=BN, num_warps=warps or 4, num_stages=2)
    return out if sk == 1 else out.sum(0).to(out_dtype)


def tq_moe_gemm(a, key, lo, hi, nlev, sorted_ids, expert_ids, ntp, num_valid, top_k, out_rows, BM,
                topk_w=None, a_per_slot=False, split=None):
    """a [T or T*topk, K] bf16; key [E, G, N], lo/hi [E, G, 26, N]; returns [out_rows (= T*topk), N] (bf16 or fp32 partial sums)."""
    E, NG, NOUT = key.shape
    WB = 16 if nlev == 9 else 24
    BN = 64
    nmb = expert_ids.shape[0]
    sk = split or 1
    hp = hi if hi is not None else lo
    if sk == 1:
        out = torch.empty(out_rows, NOUT, device=a.device, dtype=torch.bfloat16); cs = 0
    else:
        out = torch.zeros(sk, out_rows, NOUT, device=a.device, dtype=torch.float32); cs = out_rows * NOUT
    grid = (triton.cdiv(NOUT, BN), nmb, sk)
    _tq_kernel[grid](a, key, lo, hp, out, topk_w if topk_w is not None else a, sorted_ids, expert_ids, ntp,
                     0, NOUT, NG, a.stride(0), cs, NOUT, key.stride(0), lo.stride(0), num_valid, top_k,
                     NOUT, 0, NOUT, 0, NOUT,
                     NLEV=nlev, WB=WB, NLEV2=nlev, WB2=WB, TWO=False, MOE=True, MUL_W=topk_w is not None, A_PER_SLOT=a_per_slot, SPLIT_K=sk,
                     BLOCK_M=BM, BLOCK_N=BN, num_warps=4, num_stages=2)
    return out if sk == 1 else out.sum(0).to(torch.bfloat16)


def tq_moe(x, w13, w2, topk_weights, topk_ids, global_num_experts):
    """routed experts, SiLU-and-mul (gate first, up second, as vLLM w13). w13 / w2 = (key, lo, hi, nlev). x [T, H] bf16 -> [T, H]."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    T, H = x.shape; top_k = topk_ids.shape[1]
    BM = 16 if T * top_k <= 64 * global_num_experts // 16 else 32
    sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, BM, global_num_experts)
    nv = T * top_k
    k13, l13, h13, n13 = w13
    I2 = k13.shape[-1]
    sk1 = 1 if T * top_k > 64 else max(1, min(4, k13.shape[1] // 4))
    h1 = tq_moe_gemm(x, k13, l13, h13, n13, sorted_ids, expert_ids, ntp, nv, top_k, nv, BM, split=sk1)
    I = I2 // 2
    act = (torch.nn.functional.silu(h1[:, :I].float()) * h1[:, I:].float()).to(torch.bfloat16)
    k2, l2, h2, n2 = w2
    out = tq_moe_gemm(act, k2, l2, h2, n2, sorted_ids, expert_ids, ntp, nv, top_k, nv, BM,
                      topk_w=topk_weights.reshape(-1).to(torch.float32), a_per_slot=True)
    return out.view(T, top_k, H).sum(1)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""lever1.py: REDUCED-VOCAB DRAFT HEAD. The MTP drafter's logits come from rows [0, N) of the
(shared, trit TQ27) lm_head only; columns >= N are set to -inf, so the draft argmax runs over the prefix and ids >= N can never be drafted.
Verification keeps the full head (the target's compute_logits is untouched). No new weights: the kernel reads the first N output columns of
the SAME packed tensors (row stride = the full LD), with the tile / split-K choice of the FULL head (kern._cfg(M, LD_full, NG)), so every
column < N is bit-identical to the full head's value (checked by a unit gate). Knob: env TQ_DRAFT_VOCAB=N (0 / unset = off).
Same two code paths as the full head: TQ_L2 -> glue._dense_rk (split-K reduced in-kernel), else kern._tq_kernel (+ torch .sum(0))."""
import torch
import triton
from .kern import _tq_kernel, _cfg, G
from . import glue


def tq_head_slice(x, key, lo, hi, nlev, n, l2):
    """x [M, K] bf16; key [NG, LD], lo [NG, 26, LD], hi [NG, 26, LD] (TQ27) or None (TQ9) -> [M, LD] bf16 with cols >= n = -inf."""
    M, K = x.shape; NG, LD = key.shape; assert K == NG * G
    n = min(int(n), LD)
    BM, BN, sk = _cfg(M, LD, NG)                      # the FULL head's tiling (bit-identity of the kept columns)
    WB = 16 if nlev == 9 else 24
    hp = hi if hi is not None else lo
    LDH = hi.shape[-1] if hi is not None else LD
    out = torch.empty(M, LD, device=x.device, dtype=torch.bfloat16)
    ntn, ntm = triton.cdiv(n, BN), triton.cdiv(M, BM)
    if l2:
        if sk > 1:
            ws = torch.empty(sk, M, LD, device=x.device, dtype=torch.float32)
            cnt = glue._counters(("h", key.data_ptr()), ntn * 64)
        else:
            ws, cnt = out, out
        # NOUT = n (column mask / tile count), LD = full row stride of weights AND output
        glue._dense_rk[(ntn, ntm, sk)](x, key, lo, hp, out, ws, cnt, M, n, NG, x.stride(0), LD, LDH, 0, LD, ntn,
                                       NLEV=nlev, WB=WB, NLEV2=nlev, WB2=WB, TWO=False, SPLIT_K=sk, BLOCK_M=BM, BLOCK_N=BN,
                                       num_warps=4, num_stages=2)
    else:
        if sk == 1:
            buf, cs = out, 0
        else:
            buf = torch.empty(sk, M, LD, device=x.device, dtype=torch.float32); cs = M * LD
        _tq_kernel[(ntn, ntm, sk)](x, key, lo, hp, buf, x, x, x, x, x, M, n, NG, x.stride(0), cs, LD, 0, 0, 0, 1,
                                   LD, 0, LDH, 0, LD,
                                   NLEV=nlev, WB=WB, NLEV2=nlev, WB2=WB, TWO=False, MOE=False, MUL_W=False, A_PER_SLOT=False,
                                   SPLIT_K=sk, BLOCK_M=BM, BLOCK_N=BN, num_warps=4, num_stages=2)
        if sk > 1:
            out[:, :n] = buf[:, :, :n].sum(0).to(torch.bfloat16)
    if n < LD:
        out[:, n:].fill_(float("-inf"))
    return out

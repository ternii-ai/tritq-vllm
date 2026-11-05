# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""Packed trit rows for the Qwen3.8 PLE n-gram table: one row = 160 weights = 5 groups of 32 (160 % 128 != 0),
one fp16 key per group (max-anchored), codes q in [-L, L], L = (N-1)/2, N = 3 / 9 / 27 (1 / 2 / 3 trits).
Same mixed-radix word as fmt.py (5 digits per word, digit 0 most significant): word w (0..31) carries the 5 values (group k, position w),
k = 0..4, i.e. digit k of every word is group k.  x = ceil(v * 2^W / N^5), v = sum_k u_k N^(4-k), u = q + L; decode 5x: t *= N; u = t >> W; t &= mask.
Row bytes: keys 5 x fp16 (10 B) | lo 32 words (uint8 for N=3, uint16 LE for N=9/27) | hi 32 x uint8 (N=27 only).
  N=3: W=8  -> 42 B/row (2.1 b/w);  N=9: W=16 -> 74 B (3.7 b/w);  N=27: W=24 -> 106 B (5.3 b/w)."""
import torch

GR, NGR, NWR, D = 32, 5, 32, 160
WBITS = {3: 8, 9: 16, 27: 24}


def row_bytes(N):
    return 2 * NGR + (NWR if N == 3 else 2 * NWR) + (NWR if N == 27 else 0)


def quant_rows(w, N):
    """w [R, 160] (bf16/fp32) -> (q [R, 160] int32, s [R, 5] fp16) with g = 32 (key fp16, zero key -> 1)."""
    L = (N - 1) // 2
    x = w.float().reshape(-1, NGR, GR)
    s = (x.abs().amax(-1) / L).half()
    sf = s.float(); sf = torch.where(sf == 0, torch.ones_like(sf), sf)
    q = (x / sf[..., None]).round().clamp(-L, L).to(torch.int32)
    s = torch.where(s == 0, torch.ones_like(s), s)          # the key actually used (an all-zero group uses key 1)
    return q.reshape(w.shape[0], D), s


def pack_rows(q, s, N):
    """q [R, 160] int, s [R, 5] fp16 -> uint8 [R, row_bytes(N)]."""
    W = WBITS[N]; L = (N - 1) // 2; R = q.shape[0]
    u = (q.to(torch.int64) + L).reshape(R, NGR, GR)                 # [R, k, w]
    v = torch.zeros(R, NWR, dtype=torch.int64, device=q.device)
    for k in range(NGR): v = v * N + u[:, k, :]
    num = v << W; N5 = N ** 5
    x = num // N5 + (num % N5 != 0).to(torch.int64)
    assert int(x.max()) < (1 << W)
    parts = [s.to(torch.float16).contiguous().view(torch.uint8).reshape(R, 2 * NGR)]
    if N == 3:
        parts.append(x.to(torch.uint8))
    else:
        parts.append((x & 0xFFFF).to(torch.int32).to(torch.int16).contiguous().view(torch.uint8).reshape(R, 2 * NWR))
        if N == 27: parts.append(((x >> 16) & 0xFF).to(torch.uint8))
    return torch.cat(parts, 1).contiguous()


def decode_rows(b, N, out_dtype=torch.bfloat16):
    """b uint8 [..., row_bytes(N)] -> values [..., 160] = (q * key) in fp32, cast to out_dtype (graph-capturable torch ops)."""
    W = WBITS[N]; L = (N - 1) // 2; mask = (1 << W) - 1
    lead = b.shape[:-1]
    key = b[..., : 2 * NGR].contiguous().view(torch.float16).float()                       # [..., 5]
    if N == 3:
        x = b[..., 2 * NGR: 2 * NGR + NWR].to(torch.int32)
    else:
        x = b[..., 2 * NGR: 2 * NGR + 2 * NWR].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        if N == 27:
            x = x | (b[..., 2 * NGR + 2 * NWR: 2 * NGR + 3 * NWR].to(torch.int32) << 16)
    digs = []
    for _ in range(NGR):
        x = x * N; digs.append(((x >> W) - L).float()); x = x & mask
    q = torch.stack(digs, -2)                                                               # [..., 5, 32]
    return (q * key[..., None]).reshape(*lead, D).to(out_dtype)


def unpack_rows(b, N):
    """exact inverse for the gates: -> (q [R, 160] int32, s [R, 5] fp16)."""
    W = WBITS[N]; L = (N - 1) // 2; mask = (1 << W) - 1
    s = b[:, : 2 * NGR].contiguous().view(torch.float16)
    if N == 3:
        x = b[:, 2 * NGR: 2 * NGR + NWR].to(torch.int64)
    else:
        x = b[:, 2 * NGR: 2 * NGR + 2 * NWR].contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        if N == 27: x = x | (b[:, 2 * NGR + 2 * NWR:].to(torch.int64) << 16)
    digs = []
    for _ in range(NGR):
        x = x * N; digs.append((x >> W) - L); x = x & mask
    return torch.stack(digs, 1).reshape(b.shape[0], D).to(torch.int32), s

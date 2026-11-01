# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""Dense trit formats TQ9_128 / TQ27_128 (= gguf-py _TQDense, ggml-common.h), in torch, for the vLLM kernels.
One group = 128 weights along the input dim: fp16 key d + 26 mixed-radix words, 5 digits per word.
  weight j of the group -> word j % 26, digit j // 26 (digit 0 most significant); digit u = q + L, L = (N-1)/2;
  word x = ceil(v * 2^W / N^5), v = sum_i u_i N^(4-i); decode: t = x; repeat 5x: t *= N; u_i = t >> W; t &= 2^W - 1.
  N = 9 (2 trits): W = 16 -> 3.375 b/w;  N = 27 (3 trits): W = 24 (16-bit lo + 8-bit hi) -> 5.0 b/w.
Byte encoding is identical to the GGUF blocks; only the ARRANGEMENT differs (structure of arrays, transposed for coalesced loads):
  key [.., G, N_out] fp16, lo [.., G, 26, N_out] int16 (uint16 bits), hi [.., G, 26, N_out] uint8 (TQ27 only)."""
import torch

G, NW = 128, 26
WBITS = {9: 16, 27: 24}


def pack(q: torch.Tensor, d: torch.Tensor, N: int):
    """q [..., out, K] int (|q| <= L), d [..., out, K/128] fp16 -> (key, lo, hi|None) in the kernel layout. Exact (asserted by unpack)."""
    W = WBITS[N]; L = (N - 1) // 2; *lead, o, K = q.shape; ng = K // G
    assert K % G == 0 and int(q.abs().max()) <= L
    u = torch.zeros(*lead, o, ng, 5 * NW, dtype=torch.int64, device=q.device)
    u[..., :G] = (q.to(torch.int64) + L).reshape(*lead, o, ng, G)
    u = u.reshape(*lead, o, ng, 5, NW)
    v = torch.zeros(*lead, o, ng, NW, dtype=torch.int64, device=q.device)
    for i in range(5): v = v * N + u[..., i, :]
    N5 = N ** 5; num = v << W
    x = num // N5 + (num % N5 != 0).to(torch.int64)
    assert int(x.max()) < (1 << W)
    perm = list(range(len(lead))) + [len(lead) + 1, len(lead) + 2, len(lead)]            # [.., o, ng, 26] -> [.., ng, 26, o]
    lo = (x & 0xFFFF).to(torch.int32).to(torch.int16).permute(*perm).contiguous()        # wraps to int16 bits
    hi = ((x >> 16) & 0xFF).to(torch.uint8).permute(*perm).contiguous() if W == 24 else None
    key = d.to(torch.float16).transpose(-1, -2).contiguous()                             # [.., ng, o]
    return key, lo, hi


def unpack(key, lo, hi, N: int):
    """inverse of pack -> (q [..., out, K] int32, d [..., out, K/128] fp16)."""
    W = WBITS[N]; L = (N - 1) // 2; mask = (1 << W) - 1
    x = lo.to(torch.int64) & 0xFFFF
    if hi is not None: x = x | (hi.to(torch.int64) << 16)
    *lead, ng, nw, o = x.shape
    t = x; digs = []
    for i in range(5):
        t = t * N; digs.append((t >> W) - L); t = t & mask
    q = torch.stack(digs, -3).reshape(*lead, ng * 5 * NW, o)                              # [.., ng, 5, 26, o] -> j = i*26 + w
    q = q.reshape(*lead, ng, 5 * NW, o)[..., :G, :].reshape(*lead, ng * G, o)
    return q.transpose(-1, -2).to(torch.int32).contiguous(), key.transpose(-1, -2).contiguous()


def dequant_ref(key, lo, hi, N: int):
    """fp32 reference weight [..., out, K] = d * q."""
    q, d = unpack(key, lo, hi, N)
    return q.float() * d.float().repeat_interleave(G, -1)

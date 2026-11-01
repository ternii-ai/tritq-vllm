# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""sparse.py: context-aware sparse MTP draft head for the TQ lm_head. A reimplementation, for the packed-trit head, of the algorithm
of vLLM pull request #59740 by @stecasta (Apache-2.0; as carried by salient-data/qwen38-next-dgx-recipe patch 0013): same algorithm,
the gather kernel follows the PR's row-gather GEMV layout, no lines copied. The drafter's logits = exact logits of a fixed prefix [0, F) (the draft-vocabulary prefix copy, same kernel) plus, per draft token,
the J rows outside the prefix that a rank-r projection of the lm_head scores highest, with exact logits for those J rows; -inf elsewhere.
So non-English ids (CJK band >= 100k, Cyrillic / Hangul / kana >= 151,643) stay draftable. Verification keeps the full head, so only the
drafts (acceptance) can change.
Built once at load from the TQ codes: basis = top-r eigenvectors of W^T W over the real vocab (W = key * code, exact), proj = W[F:VOC] @ basis
(bf16), codes of rows [F, VOC) row-major int8 + per-(row, group) fp16 keys for the exact gather (Triton)."""
import torch
import triton
import triton.language as tl

VOC = 248070            # real vocab of the 35B (ids >= 248,070 are padding rows)


def dequant(key, lo, hi, nlev, c0, c1):
    """exact codes of rows [c0, c1) as int8 [rows, K] (TQ9: 5 base-9 digits per int16 word; digit i -> inputs i*26 + w of the group)"""
    NG = key.shape[0]; L = (nlev - 1) // 2; WB = 16 if nlev == 9 else 24; MASKW = (1 << WB) - 1
    x = lo[:, :, c0:c1].to(torch.int32) & 0xFFFF
    if WB == 24: x = x | (hi[:, :, c0:c1].to(torch.int32) << 16)
    Q = torch.zeros(c1 - c0, NG, 130, device=key.device, dtype=torch.int8)
    for i in range(5):
        x = x * nlev; q = (x >> WB) - L; x = x & MASKW
        Q[:, :, i * 26: i * 26 + 26] = q.permute(2, 0, 1).to(torch.int8)
    return Q[:, :, :128].reshape(c1 - c0, NG * 128).contiguous()


@triton.jit
def _gather_kernel(x_ptr, q_ptr, k_ptr, ids_ptr, out_ptr, J, NG: tl.constexpr, BR: tl.constexpr):
    t = tl.program_id(0); r = tl.program_id(1) * BR + tl.arange(0, BR); rm = r < J
    ids = tl.load(ids_ptr + t * J + r, mask=rm, other=0).to(tl.int64)
    cols = tl.arange(0, 128); acc = tl.zeros((BR,), dtype=tl.float32)
    for g in range(NG):
        xv = tl.load(x_ptr + t * NG * 128 + g * 128 + cols).to(tl.float32)
        q = tl.load(q_ptr + ids[:, None] * (NG * 128) + g * 128 + cols[None, :], mask=rm[:, None], other=0).to(tl.float32)
        kd = tl.load(k_ptr + ids * NG + g, mask=rm, other=0.0).to(tl.float32)
        acc += tl.sum(q * xv[None, :], axis=1) * kd
    tl.store(out_ptr + t * J + r, acc, mask=rm)


class SparseHead:
    def __init__(self, key, lo, hi, nlev, F, r, J, sc=1):
        """sc = 1: the scorer (rows x r projection) is stored in our own TQ27 format (4.75-bit codes, fp16 key per 128, 5.0 b/w) and run by
        kern.tq_linear; sc = 0: bf16 + cuBLAS (reference)."""
        dev = key.device; NG, N = key.shape; self.F, self.r, self.J, self.N, self.NG, self.sc = F, r, J, N, NG, sc
        CH = 8192; gram = torch.zeros(NG * 128, NG * 128, device=dev, dtype=torch.float64)
        self.q = torch.empty(VOC - F, NG * 128, device=dev, dtype=torch.int8)
        self.k = key[:, F:VOC].t().contiguous()                     # [rows, NG] fp16
        for c0 in range(0, VOC, CH):
            c1 = min(VOC, c0 + CH); Q = dequant(key, lo, hi, nlev, c0, c1)
            W = (Q.view(c1 - c0, NG, 128).float() * key[:, c0:c1].t().float()[:, :, None]).reshape(c1 - c0, NG * 128)
            gram += (W.t() @ W).double()
            if c1 > F:
                a = max(c0, F); self.q[a - F: c1 - F] = Q[a - c0:]
        basis = torch.linalg.eigh(gram).eigenvectors.flip(-1)[:, :r].float(); del gram
        self.basis = basis.to(torch.bfloat16).contiguous()
        proj = torch.empty(VOC - F, r, device=dev, dtype=torch.float32)
        for c0 in range(0, VOC - F, CH):
            c1 = min(VOC - F, c0 + CH)
            W = self.q[c0:c1].view(c1 - c0, NG, 128).float() * self.k[c0:c1].float()[:, :, None]
            proj[c0:c1] = W.reshape(c1 - c0, NG * 128) @ basis
        if sc:
            from .fmt import pack
            pg = proj.view(VOC - F, r // 128, 128)
            d = (pg.abs().amax(-1) / 13).clamp(min=1e-8)
            q = torch.round(pg / d[:, :, None]).clamp(-13, 13).to(torch.int8).view(VOC - F, r)
            self.sk, self.sl, self.sh = [], [], []
            for c0 in range(0, VOC - F, 65536):                         # pack in chunks (int64 temporaries)
                c1 = min(VOC - F, c0 + 65536); k_, l_, h_ = pack(q[c0:c1], d[c0:c1].half(), 27)
                self.sk.append(k_); self.sl.append(l_); self.sh.append(h_)
            self.sk = torch.cat(self.sk, 1).contiguous(); self.sl = torch.cat(self.sl, 2).contiguous(); self.sh = torch.cat(self.sh, 2).contiguous()
            self.proj_t = None; smb = (self.sk.numel() * 2 + self.sl.numel() * 2 + self.sh.numel()) / 1e6; del q, d
        else:
            self.proj_t = proj.t().contiguous().to(torch.bfloat16); smb = self.proj_t.numel() * 2 / 1e6
        del proj
        self.mb = (self.q.numel() + self.k.numel() * 2 + self.basis.numel() * 2) / 1e6 + smb; self.scorer_mb = smb

    def logits(self, x2, fixed):
        """x2 [n, K] bf16, fixed [n, F] bf16 (prefix logits) -> [n, N] bf16 target-vocab logits, -inf outside prefix + picks"""
        n = x2.shape[0]
        xb = x2 @ self.basis
        if self.sc:
            from .kern import tq_linear
            s = tq_linear(xb.contiguous(), self.sk, self.sl, self.sh, 27)     # [n, rows] approximate logits
        else:
            s = xb @ self.proj_t
        picks = s.topk(self.J, dim=-1, sorted=False).indices.to(torch.int32).contiguous()
        ex = torch.empty(n, self.J, device=x2.device, dtype=torch.float32)
        _gather_kernel[(n, triton.cdiv(self.J, 64))](x2.contiguous(), self.q, self.k, picks, ex, self.J, NG=self.NG, BR=64)
        out = torch.full((n, self.N), float("-inf"), device=x2.device, dtype=fixed.dtype)
        out[:, : self.F] = fixed
        out.scatter_(1, picks.long() + self.F, ex.to(fixed.dtype))
        return out

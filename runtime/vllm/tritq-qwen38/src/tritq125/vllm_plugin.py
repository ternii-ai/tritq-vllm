# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""vLLM 0.32 quantization method "trit" for Qwen3.8-Flash-Next (Qwen4Exp). Port of the 35B plugin (tritq,
vLLM 0.28) plus a packed-trit, disk-backed PLE n-gram table.
config.json quantization_config = {"quant_method": "trit", "nlev": 27, "ngram": {"file", "nlev", "rows", "row_bytes"}
  [, "expert_trits": {"<layer>": {"w13": "<512 chars '3'/'2'>", "w2": "..."}}]}.
  expert_trits = hot/cold routed experts: per layer and part, the level of each expert; such layers use TritMoEHCMethod (one
  packed weight set per level, checkpoint names experts.<e>.<proj>.tq_* for 3 trits and .tqc_* for 2 trits). Absent -> unchanged.
  Quantized (TQ27_128 / TQ9_128, fmt.py; Triton kernels kern.py): the vLLM modules matching QUANT_RE below + lm_head. Everything else
  (routers, in_proj_ba, hyper-connections, token embedding, MTP head, vision) stays unquantized.
  PLE n-gram table: rows.py rows in ONE raw file next to the checkpoint, np.memmap'ed (page cache, evictable, never resident as a tensor).
  The lookup mirrors vLLM's own pinned-host design (common/ngram_embedding.py Qwen4ExpPLEPinnedHostEmbedding): start_prefetch is an eager
  break in the breakable CUDA graph; it gathers the 16 rows per token on the CPU into a static GPU buffer; dequantize() decodes the packed
  rows to bf16 inside the graph.
kern.py and glue.py as in the 35B plugin (TQ27 everywhere here); env flags read once per process, default off: TQ_L1 decode MoE path, TQ_L2 in-kernel split-K reduce (both bit-identical, gated by unit tests at
the Qwen3.6-35B-A3B and Qwen3.8-Flash-Next shapes), TQ_L3 Triton bf16 GEMV for the routers / shared_expert_gate /
in_proj_ba of the target model (order-changing), TQ_HC8 hyper-connection projections of the target model at Q8_0 (hc8.py;
value-changing).
Hot/cold experts (TritMoEHCMethod, expert_trits): TQ_L1 also
serves the two-level expert sets (glue.tq_moe_decode with level groups); invalid expert ids are masked to weight 0 (no clamp).
Draft vocabulary option: TQ_DRAFT_VOCAB=N (default 0 = off) -> the MTP drafter's compute_logits uses rows
[0, N) of the shared trit lm_head (lever1.tq_head_slice: same packed tensors, kept columns bit-identical to the full head, cols >= N = -inf);
the target's compute_logits (verification) keeps the full head.
PLE n-gram row lookup options (all default off here; the release package turns TQ_NG_PF on, see __init__.py):
TQ_NG_HOT=<dir> resident hot-row set (_gather_rows_hot), TQ_NG_PF=1 CPU page prefetch in stream order (ngpf.c host function),
TQ_NG_RANDOM=1 madvise(MADV_RANDOM) on the table mapping, TQ_NG_PFSTAT=1 count non-resident pages, TQ_NG_DUMP=<file> log the
looked-up ids (host function, works in CUDA graphs).
(An "earlier issue" option was dropped: the PLE sits on layers[1] and its lookup already starts before layer 0, the earliest point
within a step at which its input ids exist.)
Skinny-GEMM plan fill option: TQ_MX_SKINNY=1 (see _patch_skinny_fill; derived from salient-data/qwen38-next-dgx-recipe patch 0011,
Copyright 2026 Salient Data, Apache-2.0)."""
import os, re
import numpy as np
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.logger import init_logger
from . import kern, rows as RW, glue, hc8, lever1, tk2

logger = init_logger(__name__)
G, NW = 128, 26
QUANT_RE = re.compile(r"layers\.\d+\.(linear_attn\.(in_proj_qkvz|out_proj)|self_attn\.(qkv_proj|o_proj)|"
                      r"mlp\.shared_expert\.(gate_up_proj|down_proj)|mlp\.experts|ple\.kv_proj)$")


_L1, _L2, _L3, _HC8 = glue.flag("TQ_L1"), glue.flag("TQ_L2"), glue.flag("TQ_L3"), glue.flag("TQ_HC8")
# v2 kernels from tk2.py, each behind its own flag (the release package turns K_DEC / K_MOE / K_HC on, see __init__.py).
# Every choice below is from per-shape microbenchmarks; all are bit-identical
# to the kernel they replace (BN / BLOCK_M / warps / stages / decode are value-neutral; split-K ranges and reduction orders are kept).
#   TQ_K_DEC=1  dense trit linears -> tk2.tq_linear2 with a per-shape, per-row-band tile (_K_TILES); other shapes unchanged (needs TQ_L2)
#   TQ_K_MOE=1  routed experts with 64 < T * top_k and T <= 32 rows -> tk2.tq_moe_batched (replaces the sort-based fallback there;
#               <= 64 slots stays on L1, > 32 rows stays on the fallback: both measured faster there)
#   TQ_K_HC=1   HC8 down+inject / final-mixer down (K 10240) at > 16 rows -> tk2.q8_linear2, BLOCK_M 16, today's split kept
#   TQ_K_PDL=1  programmatic dependent launch on the v2 dense / HC kernels (value-neutral; measured no gain in the chain microbench)
_K_DEC, _K_MOE, _K_HC, _K_PDL = glue.flag("TQ_K_DEC"), glue.flag("TQ_K_MOE"), glue.flag("TQ_K_HC"), glue.flag("TQ_K_PDL")
# (K, N) -> [(max rows, (BLOCK_N, warps, stages)), ...] ascending; rows above the last band -> today's kernel
_K_TILES = {(2560, 16384): [(16, (256, 4, 2)), (64, (128, 8, 3))],          # GDN in_proj_qkvz
            (6144, 2560): [(16, None), (32, (64, 4, 3)), (64, (128, 8, 3))],  # GDN out_proj / attn o_proj
            (2560, 1280): [(32, (64, 4, 3)), (64, (128, 8, 3))],              # shared gate_up
            (640, 2560): [(32, (64, 4, 3)), (64, (64, 4, 2))],                # shared down
            (2560, 248320): [(64, (128, 4, 3))]}                              # lm_head (verify; the draft slice is lever1's kernel)
_K_ATTN_QKV = [(16, (256, 4, 2)), (64, (128, 8, 3))]                          # attn qkv (2560 -> N_qkv), N read from the weight
_K_SEEN = set()


def _k_tile(K, N, M):
    bands = _K_TILES.get((K, N))
    if bands is None and K == 2560 and 12000 < N < 16000:
        bands = _K_ATTN_QKV
    if bands is None:
        if (K, N) not in _K_SEEN and not torch.cuda.is_current_stream_capturing():
            _K_SEEN.add((K, N)); logger.warning("trit v2 kernels: dense shape K %d N %d has no v2 tile -> today's kernel", K, N)
        return False, None
    for mmax, t in bands:
        if M <= mmax:
            return (t is not None), t
    return False, None
_L1_DEBUG = glue.flag("TQ_L1_DEBUG"); _SEEN = set()                # diagnostic only: routing shapes / id range outside graph capture
_DRAFT_N = int(os.environ.get("TQ_DRAFT_VOCAB", "0") or 0)          # draft vocabulary: draft-head vocab prefix (0 = off)
# Sparse context-aware draft head (sparse.py): TQ_SPARSE="F,r,J,sc" -> drafter logits = exact prefix [0, F) (the draft-vocab slice kernel)
# + J rows per draft token picked by a rank-r TQ27 scorer, exact logits for those rows, -inf elsewhere.
_SP = tuple(int(v) for v in os.environ.get("TQ_SPARSE", "").split(",")) if os.environ.get("TQ_SPARSE") else None
if _SP: _DRAFT_N = _SP[0]
_SPARSE = [None]
_IN_DRAFT = [False]                                                  # set only inside Qwen4ExpMTP.compute_logits (eager, drafter)
_DRAFT_CALLS = [0]


def _tq_linear_op(x: torch.Tensor, key: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, nlev: int) -> torch.Tensor:
    if _K_DEC and _L2:
        use, t = _k_tile(key.shape[0] * 128, key.shape[1], x.shape[0])
        if use:
            return tk2.tq_linear2(x, key, lo, hi if nlev == 27 else None, nlev, bn=t[0], warps=t[1], stages=t[2], dec=1, pdl=_K_PDL)
    lin = glue.tq_linear_rk if _L2 else kern.tq_linear              # TQ_L2: split-K reduced inside the kernel (bit-identical)
    return lin(x, key, lo, hi if nlev == 27 else None, nlev)


def _tq_linear_fake(x, key, lo, hi, nlev):
    return torch.empty(x.shape[0], key.shape[1], dtype=x.dtype, device=x.device)


direct_register_custom_op("tq_linear", _tq_linear_op, mutates_args=[], fake_impl=_tq_linear_fake)


def _tq_head_draft_op(x: torch.Tensor, key: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, nlev: int, n: int) -> torch.Tensor:
    return lever1.tq_head_slice(x, key, lo, hi if nlev == 27 else None, nlev, n, _L2)


direct_register_custom_op("tq_head_draft", _tq_head_draft_op, mutates_args=[], fake_impl=lambda x, key, lo, hi, nlev, n:
                          torch.empty(x.shape[0], key.shape[1], dtype=x.dtype, device=x.device))


def _gemv_op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return glue.bf16_gemv(x, w)


def _gemv_fake(x, w):
    return torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)


direct_register_custom_op("tq_gemv_bf16", _gemv_op, mutates_args=[], fake_impl=_gemv_fake)


class Bf16GemvMethod(UnquantizedLinearMethod):
    """bf16 weights as stored; decode-size inputs (M <= 8, no bias) go through one Triton GEMV (TQ_L3) instead of vLLM's SM121
    skinny-GEMM plan / cuBLAS"""
    def apply(self, layer, x, bias=None):
        x2 = x.reshape(-1, x.shape[-1])
        if bias is None and x2.shape[0] <= 8 and layer.weight.dtype == torch.bfloat16 and x2.dtype == torch.bfloat16:
            y = torch.ops.vllm.tq_gemv_bf16(x2.contiguous(), layer.weight)
            return y.reshape(*x.shape[:-1], y.shape[-1])
        return super().apply(layer, x, bias)


def _hc8_op(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, silu_n: int, hc: int) -> torch.Tensor:
    if _K_HC and x.shape[0] > 16 and q.shape[0] == 10240:
        _, _, sk = hc8._cfg(x.shape[0], q.shape[1], q.shape[0] // 32, None)     # today's split at this M = today's numerics
        return tk2.q8_linear2(x, q, s, silu_n=silu_n, hc=hc, dec=1, pdl=_K_PDL, bm=16, bn=64, split=sk, warps=8, stages=3, gpi=2)
    return hc8.q8_linear(x, q, s, silu_n=silu_n, hc=hc)


def _hc8_fake(x, q, s, silu_n, hc):
    return torch.empty(x.shape[0], q.shape[1], dtype=x.dtype, device=x.device)


direct_register_custom_op("tq_hc8", _hc8_op, mutates_args=[], fake_impl=_hc8_fake)
_HC8_FUSE = os.environ.get("TQ_HC8_FUSE", "1") == "1"         # fused hc_silu epilogue on the down / inject projection (default on with HC8)


class HC8Method(UnquantizedLinearMethod):
    """TQ_HC8: a bf16 HC projection (built with quant_config=None) re-quantized after loading to Q8_0 (hc8.quantize); the bf16 weight is
    freed. nreal: rows beyond it (the merged down / inject module's 16-row alignment pad, never loaded) are zeroed before quantizing."""
    def __init__(self, nreal=None):
        self.nreal = nreal

    def process_weights_after_loading(self, layer):
        w = layer.weight.data
        if self.nreal is not None and self.nreal < w.shape[0]:
            w[self.nreal:].zero_()
        q, s = hc8.quantize(w)
        layer.register_buffer("hc8_q", q, persistent=False); layer.register_buffer("hc8_s", s, persistent=False)
        layer.hc8_n = w.shape[0]
        layer.weight = Parameter(torch.empty(0, w.shape[1], dtype=w.dtype, device=w.device), requires_grad=False)
        torch.cuda.empty_cache()

    def apply(self, layer, x, bias=None, silu_n=0, hc=1):
        x2 = x.reshape(-1, x.shape[-1])
        if x2.stride(1) != 1: x2 = x2.contiguous()
        y = torch.ops.vllm.tq_hc8(x2, layer.hc8_q, layer.hc8_s, silu_n, hc)
        if bias is not None: y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])


_MTP_FP8 = os.environ.get("TQ_MTP_FP8", "0") == "1"


def _quantized(prefix: str) -> bool:
    if "mtp" in prefix or "visual" in prefix:
        return False
    return prefix.endswith("lm_head") or QUANT_RE.search(prefix) is not None


@register_quantization_config("trit")
class TritConfig(QuantizationConfig):
    def __init__(self, nlev: int, ngram: dict | None, expert_trits: dict | None = None):
        super().__init__()
        self.nlev, self.ngram, self.expert_trits = nlev, ngram, expert_trits
        self.n_quant = 0
        self._fp8 = None

    def __repr__(self):
        return f"TritConfig(nlev={self.nlev}, ngram={self.ngram})"

    @classmethod
    def get_name(cls):
        return "trit"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls(int(config["nlev"]), config.get("ngram"), config.get("expert_trits"))

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        q = _quantized(prefix)
        # TQ_MTP_FP8=1: the MTP draft head's linears and routed experts quantised ONLINE to FP8 per tensor by
        # vLLM 0.32's online methods (bf16 checkpoint weights -> e4m3 at load, dynamic activation scales). Drafts only: verification uses the trit target.
        # Excluded: the MTP head's lm_head (trit already), its routers / shared_expert_gate (tiny), hyper-connections (not LinearBase here).
        if _MTP_FP8 and "mtp" in prefix and not prefix.endswith("lm_head") and not prefix.endswith((".gate", "shared_expert_gate")) \
                and (isinstance(layer, RoutedExperts) or isinstance(layer, LinearBase)):
            from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod, Fp8PerTensorOnlineMoEMethod
            self.n_mtp_fp8 = getattr(self, "n_mtp_fp8", 0) + 1
            logger.info("trit: MTP FP8 per-tensor (online) %s", prefix)
            return Fp8PerTensorOnlineMoEMethod(moe=layer.moe_config) if isinstance(layer, RoutedExperts) else Fp8PerTensorOnlineLinearMethod()
        if isinstance(layer, RoutedExperts):
            if q:
                self.n_quant += 1
                m = re.search(r"layers\.(\d+)\.mlp\.experts$", prefix)
                if self.expert_trits and m and m.group(1) in self.expert_trits:
                    return TritMoEHCMethod(self.expert_trits[m.group(1)], layer.moe_config)
                return TritMoEMethod(self.nlev, self.nlev, layer.moe_config)
            logger.info("trit: unquantized experts %s", prefix)
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
            return UnquantizedFusedMoEMethod(layer.moe_config)
        if isinstance(layer, ParallelLMHead) or isinstance(layer, LinearBase):
            if q:
                self.n_quant += 1
                return TritLinearMethod(self.nlev)
            if isinstance(layer, ParallelLMHead):
                from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
                return UnquantizedEmbeddingMethod()
            if _L3 and prefix.endswith("in_proj_ba") and "mtp" not in prefix:
                return Bf16GemvMethod()
            return UnquantizedLinearMethod()
        return None


class TritLinearMethod(LinearMethodBase):
    def __init__(self, nlev):
        self.nlev = nlev

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype,
                       **extra_weight_attrs):
        K = input_size_per_partition; NO = sum(output_partition_sizes); assert K % G == 0, (K, layer)
        ng = K // G; wl = extra_weight_attrs.get("weight_loader")
        key = Parameter(torch.empty(ng, NO, dtype=torch.float16), requires_grad=False)
        lo = Parameter(torch.empty(ng, NW, NO, dtype=torch.int16), requires_grad=False)
        set_weight_attrs(key, {"output_dim": 1, "weight_loader": wl}); set_weight_attrs(lo, {"output_dim": 2, "weight_loader": wl})
        layer.register_parameter("tq_key", key); layer.register_parameter("tq_lo", lo)
        if self.nlev == 27:
            hi = Parameter(torch.empty(ng, NW, NO, dtype=torch.uint8), requires_grad=False)
            set_weight_attrs(hi, {"output_dim": 2, "weight_loader": wl}); layer.register_parameter("tq_hi", hi)
        layer.tq_nlev = self.nlev

    def process_weights_after_loading(self, layer):
        if self.nlev != 27:
            layer.tq_hi_t = layer.tq_lo
        if _SP and getattr(layer, "tq_is_head", False) and _SPARSE[0] is None:
            from .sparse import SparseHead, VOC
            assert layer.tq_key.shape[1] >= VOC, (layer.tq_key.shape, VOC)
            _SPARSE[0] = SparseHead(layer.tq_key, layer.tq_lo, getattr(layer, "tq_hi", None), self.nlev, *_SP)
            logger.warning("trit sparse draft head F,r,J(,sc)=%s, VOC %d, %.1f MB extra (scorer %.1f MB)", _SP, VOC, _SPARSE[0].mb, _SPARSE[0].scorer_mb)

    def apply(self, layer, x, bias=None):
        shp = x.shape; x2 = x.reshape(-1, shp[-1]).contiguous()
        hi = layer.tq_hi if self.nlev == 27 else layer.tq_hi_t
        if _DRAFT_N and _IN_DRAFT[0] and getattr(layer, "tq_is_head", False):
            _DRAFT_CALLS[0] += 1
            if _DRAFT_CALLS[0] <= 3:
                logger.warning("trit draft-vocab: draft head slice N=%d, x %s", _DRAFT_N, tuple(x2.shape))
            y = torch.ops.vllm.tq_head_draft(x2, layer.tq_key, layer.tq_lo, hi, self.nlev, _DRAFT_N)
            if _SPARSE[0] is not None:
                y = _SPARSE[0].logits(x2, y[:, :_DRAFT_N])
        else:
            y = torch.ops.vllm.tq_linear(x2, layer.tq_key, layer.tq_lo, hi, self.nlev)
        if bias is not None: y = y + bias
        return y.reshape(*shp[:-1], y.shape[-1])


class TritMoEMethod(FusedMoEMethodBase):
    def __init__(self, n13, n2, moe):
        super().__init__(moe)
        self.n13, self.n2 = n13, n2

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
        E, H, I = num_experts, hidden_size, intermediate_size_per_partition
        def mk(name, ng, no, nlev, w13):
            ps = {"key": torch.empty(E, ng, no, dtype=torch.float16), "lo": torch.empty(E, ng, NW, no, dtype=torch.int16)}
            if nlev == 27: ps["hi"] = torch.empty(E, ng, NW, no, dtype=torch.uint8)
            for k, t in ps.items():
                p = Parameter(t, requires_grad=False); layer.register_parameter(f"{name}_tq_{k}", p)
                set_weight_attrs(p, {"weight_loader": self._loader(w13, I)})
        mk("w13", H // G, 2 * I, self.n13, True); mk("w2", I // G, H, self.n2, False)

    @staticmethod
    def _loader(w13, I):
        def load(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            dst = param.data[expert_id]
            if w13:
                assert shard_id in ("w1", "w3"), shard_id
                dst = dst.narrow(-1, 0 if shard_id == "w1" else I, I)
            if dst.dim() == 3: loaded_weight = loaded_weight.reshape(dst.shape)     # words stored flat [G*26, out]
            assert dst.shape == loaded_weight.shape, (weight_name, dst.shape, loaded_weight.shape)
            dst.copy_(loaded_weight)
            return True if return_success else None
        return load

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        w13 = (layer.w13_tq_key, layer.w13_tq_lo, getattr(layer, "w13_tq_hi", None), self.n13)
        w2 = (layer.w2_tq_key, layer.w2_tq_lo, getattr(layer, "w2_tq_hi", None), self.n2)
        if _L1_DEBUG and not torch.cuda.is_current_stream_capturing():
            key_ = (tuple(x.shape), tuple(topk_ids.shape), str(topk_ids.dtype))
            if key_ not in _SEEN:
                _SEEN.add(key_)
                print(f"TQ_L1_DEBUG x {key_[0]} ids {key_[1]} {key_[2]} min {int(topk_ids.min())} max {int(topk_ids.max())} E {layer.global_num_experts}",
                      flush=True)
        if _K_MOE and topk_ids.shape[0] <= 32 and topk_ids.shape[0] * topk_ids.shape[1] > 64:
            return tk2.tq_moe_batched(x.contiguous(), w13, w2, topk_weights, topk_ids.to(torch.int32))
        if _L1 and topk_ids.shape[0] * topk_ids.shape[1] <= 64 and isinstance(self.n13, int) and isinstance(self.n2, int):
            return glue.tq_moe_decode(x.contiguous(), w13, w2, topk_weights, topk_ids.to(torch.int32))     # bit-identical (unit gates)
        return kern.tq_moe(x.contiguous(), w13, w2, topk_weights, topk_ids.to(torch.int32), layer.global_num_experts)


SUFFIX = {27: "tq", 9: "tqc"}          # checkpoint / parameter name per expert level (hot/cold)


class TritMoEHCMethod(FusedMoEMethodBase):
    """hot/cold routed experts: per part (w13 = gate_up, w2 = down) the experts are split by level; each level has its own packed
    weight set [E_level, ...] (TQ27 params w13_tq_*, TQ9 params w13_tqc_*) and an int32 map global expert id -> local index (-1 = other
    level). Loading: vLLM maps experts.<e>.gate_proj.tqc_key -> w13_tqc_key with expert_id = e; the loader puts it at the local index and
    asserts the expert belongs to that level. Compute: kern.tq_moe_hc (one alignment, one launch per level per GEMM)."""
    def __init__(self, trits: dict, moe):
        super().__init__(moe)
        self.trits = trits

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
        E, H, I = num_experts, hidden_size, intermediate_size_per_partition
        layer.tq_hc_groups = {}
        for part, ng, no in (("w13", H // G, 2 * I), ("w2", I // G, H)):
            lv = [3 ** int(c) for c in self.trits[part]]
            assert len(lv) == E and set(lv) <= {9, 27}, (part, len(lv), E, set(lv))
            groups = []
            for nlev in (27, 9):
                ids = [e for e in range(E) if lv[e] == nlev]
                if not ids: continue
                emap = torch.full((E,), -1, dtype=torch.int32)
                emap[torch.tensor(ids)] = torch.arange(len(ids), dtype=torch.int32)
                ne = len(ids); sfx = SUFFIX[nlev]
                ps = {"key": torch.empty(ne, ng, no, dtype=torch.float16), "lo": torch.empty(ne, ng, NW, no, dtype=torch.int16)}
                if nlev == 27: ps["hi"] = torch.empty(ne, ng, NW, no, dtype=torch.uint8)
                for k, t in ps.items():
                    p = Parameter(t, requires_grad=False); layer.register_parameter(f"{part}_{sfx}_{k}", p)
                    set_weight_attrs(p, {"weight_loader": self._loader(part == "w13", I, emap.tolist(), nlev, f"{part}_{sfx}_{k}")})
                layer.register_buffer(f"{part}_{sfx}_emap", emap, persistent=False)
                groups.append((sfx, nlev))
            layer.tq_hc_groups[part] = groups

    @staticmethod
    def _loader(w13, I, emap, nlev, pname):
        def load(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            li = emap[expert_id]
            assert li >= 0, f"expert {expert_id} is not a {nlev}-level expert but the checkpoint has {weight_name} -> {pname}"
            dst = param.data[li]
            if w13:
                assert shard_id in ("w1", "w3"), shard_id
                dst = dst.narrow(-1, 0 if shard_id == "w1" else I, I)
            if dst.dim() == 3: loaded_weight = loaded_weight.reshape(dst.shape)     # words stored flat [G*26, out]
            assert dst.shape == loaded_weight.shape, (weight_name, dst.shape, loaded_weight.shape)
            dst.copy_(loaded_weight)
            return True if return_success else None
        return load

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer):
        # rebuild the id maps on the weights' device (create_weights may run under a meta / CPU device context)
        for part, groups in layer.tq_hc_groups.items():
            lv = [3 ** int(c) for c in self.trits[part]]
            for sfx, nlev in groups:
                ids = [e for e in range(len(lv)) if lv[e] == nlev]
                emap = torch.full((len(lv),), -1, dtype=torch.int32); emap[torch.tensor(ids)] = torch.arange(len(ids), dtype=torch.int32)
                dev = getattr(layer, f"{part}_{sfx}_key").device
                setattr(layer, f"{part}_{sfx}_emap", emap.to(dev))

    def _groups(self, layer, part):
        return [(getattr(layer, f"{part}_{s}_key"), getattr(layer, f"{part}_{s}_lo"), getattr(layer, f"{part}_{s}_hi", None), n,
                 getattr(layer, f"{part}_{s}_emap")) for s, n in layer.tq_hc_groups[part]]

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        g13, g2 = self._groups(layer, "w13"), self._groups(layer, "w2")
        if _K_MOE and topk_ids.shape[0] <= 32 and topk_ids.shape[0] * topk_ids.shape[1] > 64 and len(g13) <= 2 and len(g2) <= 2:
            return tk2.tq_moe_batched(x.contiguous(), g13, g2, topk_weights, topk_ids.to(torch.int32))
        if _L1 and topk_ids.shape[0] * topk_ids.shape[1] <= 64 and len(g13) <= 2 and len(g2) <= 2:
            # merged tree: decode MoE over the two-level sets (bit-identical per slot to tq_moe_hc; invalid ids -> weight 0)
            return glue.tq_moe_decode(x.contiguous(), g13, g2, topk_weights, topk_ids.to(torch.int32))
        return kern.tq_moe_hc(x.contiguous(), g13, g2, topk_weights, topk_ids.to(torch.int32), layer.global_num_experts)


# ------------------------------------------------------------------------------------------------ PLE n-gram table
import triton
import triton.language as tl


@triton.jit
def _gather_rows(bases_ptr, ids_ptr, out_ptr, part_rows, nrows, RB: tl.constexpr, BD: tl.constexpr):
    """out[r, :RB] = row ids[r] of a table split into equal parts of part_rows rows, part j at host address bases[j] (memmap)."""
    r = tl.program_id(0).to(tl.int64)
    i = tl.load(ids_ptr + r).to(tl.int64)
    i = tl.minimum(tl.maximum(i, 0), nrows - 1)
    j = i // part_rows
    src = tl.load(bases_ptr + j).to(tl.pointer_type(tl.uint8)) + (i - j * part_rows) * RB
    o = tl.arange(0, BD); m = o < RB
    tl.store(out_ptr + r * RB + o, tl.load(src + o, mask=m), mask=m)


@triton.jit
def _gather_rows_hot(bases_ptr, ids_ptr, out_ptr, part_rows, nrows, bm_ptr, pf_ptr, hot_ptr, RB: tl.constexpr, BD: tl.constexpr):
    """(n-gram lookup) = _gather_rows, but rows in a resident hot set come from hot_ptr: bit i of bm (uint32 words, stored int32)
    marks row i hot; its slot = pf[i >> 5] + popcount(bm[i >> 5] & ((1 << (i & 31)) - 1)) (pf = exclusive prefix of per-word popcounts;
    hot rows stored in ascending row id). Misses read the memmap exactly as _gather_rows does; a hit never touches the memmap (masked)."""
    r = tl.program_id(0).to(tl.int64)
    i = tl.load(ids_ptr + r).to(tl.int64)
    i = tl.minimum(tl.maximum(i, 0), nrows - 1)
    w = i >> 5
    b = (i & 31).to(tl.uint32)
    word = tl.load(bm_ptr + w).to(tl.uint32, bitcast=True)
    hit = ((word >> b) & 1) != 0
    x = word & ((tl.full((), 1, tl.uint32) << b) - 1)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    pc = ((x * 0x01010101) >> 24).to(tl.int64)
    slot = tl.load(pf_ptr + w).to(tl.int64) + pc
    j = i // part_rows
    o = tl.arange(0, BD); m = o < RB
    vh = tl.load(hot_ptr + slot * RB + o, mask=m & hit, other=0)
    src = tl.load(bases_ptr + j, mask=hit == 0, other=0).to(tl.pointer_type(tl.uint8)) + (i - j * part_rows) * RB
    vm = tl.load(src + o, mask=m & (hit == 0), other=0)
    tl.store(out_ptr + r * RB + o, tl.where(hit, vh, vm), mask=m)


def ng_hot_tables(ids, nrows):
    """sorted unique hot row ids (int64 numpy) -> (bitmap int32 [ceil(nrows/32)], exclusive prefix int32 [same]) for _gather_rows_hot."""
    ids = np.asarray(ids, dtype=np.int64)
    assert ids.ndim == 1 and (ids.size == 0 or (ids[0] >= 0 and ids[-1] < nrows and np.all(np.diff(ids) > 0))), "hot ids: sorted unique in range"
    nw = (nrows + 31) // 32
    w = ids >> 5
    bm = np.zeros(nw, dtype=np.uint64)
    np.add.at(bm, w, np.left_shift(np.uint64(1), (ids & 31).astype(np.uint64)))      # unique bits -> add == or
    cnt = np.bincount(w, minlength=nw).astype(np.int64)
    pf = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    assert pf[-1] + cnt[-1] == ids.size < 2 ** 31
    return bm.astype(np.uint32).view(np.int32), pf.astype(np.int32)


_NG_HOT, _NG_DUMP, _NG_PF, _NG_STAT = os.environ.get("TQ_NG_HOT", ""), os.environ.get("TQ_NG_DUMP", ""), glue.flag("TQ_NG_PF"), glue.flag("TQ_NG_PFSTAT")
_NG_RANDOM = glue.flag("TQ_NG_RANDOM")
_NG_PF_RING, _NG_PF_CAP = 32, 96            # eager ring slots, permanent slots for captured lookups (one per captured graph)


def _ngpf_lib():
    """(n-gram lookup) build + load ngpf.c (CPU page prefetch as a CUDA host function) -> ctypes lib."""
    import ctypes, hashlib, subprocess, tempfile
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngpf.c")
    tag = hashlib.md5(open(src, "rb").read()).hexdigest()[:12]
    d = os.path.join(os.environ.get("HOME", tempfile.gettempdir()), ".cache", "tritq125"); os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"ngpf_{tag}.so")
    if not os.path.exists(so):
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", tmp, src, "-ldl"]); os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    V, I, I64, U64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_uint64
    lib.ngpf_init.restype = V; lib.ngpf_init.argtypes = [ctypes.c_char_p, U64, I64, I64, I, I, I, I, ctypes.c_char_p, I]
    lib.ngpf_launch.restype = I; lib.ngpf_launch.argtypes = [V, V, I, U64, I]
    lib.ngpf_stats.argtypes = [V, V]
    return lib


def _ple_classes():
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.config import get_current_vllm_config
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding, Qwen4ExpPLEEmbeddingMethod

    class TritPLEMethod(Qwen4ExpPLEEmbeddingMethod):
        """packed trit rows; the table itself is a memmap owned by the embedding module, not a parameter."""
        def __init__(self, meta):
            self.meta = meta

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype,
                           **extra_weight_attrs):
            # a 1-row buffer named "weight" (logged / device-checked by the model); NOT a parameter, so the loader expects no tensor
            layer.register_buffer("weight", torch.zeros(1, RW.row_bytes(self.meta["nlev"]), dtype=torch.uint8), persistent=False)

        def embedding(self, layer, input_):
            raise RuntimeError("trit PLE rows are looked up through start_prefetch()")

        def dequantize(self, layer, embeddings, output_dtype):
            T = embeddings.shape[0]
            rb = RW.row_bytes(self.meta["nlev"])
            vals = RW.decode_rows(embeddings.reshape(T, -1, rb), self.meta["nlev"], output_dtype)
            return vals.reshape(T, -1)

    class RowsPLEEmbedding(Qwen4ExpPLEEmbedding):
        """PLE table read from disk through the page cache (np.memmap), never resident: trit rows (one raw file, TritPLEMethod) or the
        checkpoint's own FP8 rows straight from its safetensors shards (any other method, TRIT_PLE_MMAP=1). Lookup = a Triton gather that
        dereferences the memmap directly from the GPU (pageable-memory access), on a side stream started one layer early (vLLM's prefetch
        hook), into a static GPU buffer; the method's dequantize() runs in the graph."""
        supports_prefetch = True

        def __init__(self, num_embeddings, embedding_dim, *, params_dtype, padding_size, prefix, embedding_method,
                     num_ngram_heads=1, max_total_tokens=0, data_parallel_rank=0, reduce_results=True):
            super().__init__(num_embeddings, embedding_dim, params_dtype=params_dtype, padding_size=padding_size, prefix=prefix,
                             embedding_method=embedding_method, num_ngram_heads=num_ngram_heads, max_total_tokens=max_total_tokens,
                             data_parallel_rank=data_parallel_rank, reduce_results=reduce_results)
            assert self.tp_size == 1, "disk-backed PLE rows: TP=1 only"
            root = get_current_vllm_config().model_config.model
            if isinstance(embedding_method, TritPLEMethod):
                m = embedding_method.meta; path = os.path.join(root, m["file"])
                self.rb, self.nrows = int(m["row_bytes"]), int(m["rows"]); self.out_dtype = torch.uint8
                assert os.path.getsize(path) == self.nrows * self.rb, (path, os.path.getsize(path))
                self.parts = [(0, np.memmap(path, dtype=np.uint8, mode="r", shape=(self.nrows, self.rb)))]; self.pf_path = path
            else:
                self.pf_path = None
                self.parts, self.rb, self.out_dtype = _safetensors_rows(root, prefix)
                self.nrows = self.parts[-1][0] + self.parts[-1][1].shape[0]
                # the checkpoint shards are not copied anywhere: make the loader's shard writes no-ops
                noop = lambda *a, **k: None
                if hasattr(self.weight, "_weight_loader"): self.weight._weight_loader = noop     # BasevLLMParameter (property)
                else: self.weight.weight_loader = noop
            self.starts = np.array([p[0] for p in self.parts], dtype=np.int64)
            self.nh = num_ngram_heads; self.max_rows = max_total_tokens
            self.buf = None
            logger.info("disk-backed PLE rows %s: %d rows x %d B in %d file(s), dtype %s", prefix, self.nrows, self.rb, len(self.parts),
                        self.out_dtype)

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            return torch.zeros(1, embedding_dim, dtype=dtype)       # placeholder (FP8 method); the table stays on disk

        def _alloc(self, dev):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("PLE row buffer must be allocated before cudagraph capture")
            self.buf = torch.empty(self.max_rows, self.nh, self.rb, dtype=torch.uint8, device=dev)
            # the GPU reads the memmapped file DIRECTLY (GB10: cudaDevAttrPageableMemoryAccess = 1, host page tables): no CPU sync, no
            # pinned copy, graph-capturable; page faults fill from the page cache / NVMe
            self.bases = torch.tensor([mm.ctypes.data for _, mm in self.parts], dtype=torch.int64, device=dev)
            sizes = [mm.shape[0] for _, mm in self.parts]
            self.part_rows = sizes[0]
            assert all(x == self.part_rows for x in sizes[:-1]) and sizes[-1] <= self.part_rows, "row parts must be equal-sized"
            self.stream = torch.cuda.Stream(device=dev)
            if _NG_RANDOM:
                # (n-gram lookup) TQ_NG_RANDOM: madvise(MADV_RANDOM) on the table mapping(s): a page fault reads only its own 4 KiB page instead
                # of the default read-around window (the rows are hash-scattered, so the neighbours are never the next rows needed)
                import mmap as _mm
                for _, mmp in self.parts: mmp._mmap.madvise(_mm.MADV_RANDOM)
                logger.info("tritq n-gram: TQ_NG_RANDOM on %d mapping(s)", len(self.parts))
            self.hot = None
            if _NG_HOT:
                # (n-gram lookup) TQ_NG_HOT=<dir> with ids.npy (sorted unique int64 row ids) + rows.bin (those rows, ascending id, rb bytes
                # each, = the table's bytes): resident on the GPU; misses fall back to the memmap
                hid = np.load(os.path.join(_NG_HOT, "ids.npy"))
                hrows = np.fromfile(os.path.join(_NG_HOT, "rows.bin"), dtype=np.uint8)
                assert hrows.size == hid.size * self.rb, (hrows.size, hid.size, self.rb)
                bm, pf = ng_hot_tables(hid, self.nrows)
                self.hot = (torch.from_numpy(bm).to(dev), torch.from_numpy(pf).to(dev), torch.from_numpy(hrows).to(dev))
                logger.info("tritq n-gram: hot PLE rows %s: %d rows (%.3f GiB rows + %.3f GiB index) resident on %s", _NG_HOT, hid.size,
                            hrows.size / 2 ** 30, (bm.nbytes + pf.nbytes) / 2 ** 30, dev)
            self.pf = None
            if _NG_PF or _NG_STAT or _NG_DUMP:
                # (n-gram lookup) per lookup, on the side stream: ids -> pinned slot (D2H) -> host function ngpf.c (TQ_NG_PF: fadvise WILLNEED
                # every page of the step's rows at once, then touch them through the same mapping; TQ_NG_PFSTAT: count non-resident
                # pages first; TQ_NG_DUMP: append the ids) -> the unchanged gather. Captured into CUDA graphs as memcpy + host nodes.
                assert len(self.parts) == 1 and self.pf_path, "TQ_NG_PF/PFSTAT/DUMP: single-file (trit) table only"
                self.pf_lib = _ngpf_lib(); self.pf_maxn = self.max_rows * self.nh
                self.pf = self.pf_lib.ngpf_init(self.pf_path.encode(), self.parts[0][1].ctypes.data, self.nrows, self.rb, self.pf_maxn, self.nh,
                                                int(_NG_PF), int(_NG_STAT), _NG_DUMP.encode(), _NG_PF_RING + _NG_PF_CAP)
                assert self.pf, "ngpf_init failed"
                self.pf_k = 0; self.pf_cap = _NG_PF_RING
                logger.info("tritq n-gram: host-function path on: prefetch %d stat %d dump %s (%d ids max)", int(_NG_PF), int(_NG_STAT),
                            _NG_DUMP or "-", self.pf_maxn)

        @eager_break_during_capture
        def start_prefetch(self, ngram_ids):
            if self.buf is None: self._alloc(ngram_ids.device)
            T = ngram_ids.shape[0]; ids = ngram_ids.reshape(-1)
            self.stream.wait_stream(torch.cuda.current_stream())
            ids.record_stream(self.stream)
            with torch.cuda.stream(self.stream):
                if self.pf is not None:
                    assert ids.dtype == torch.int64 and ids.is_contiguous()
                    if torch.cuda.is_current_stream_capturing():          # graph: a permanent slot per captured lookup (n fixed in the graph)
                        k = self.pf_cap; self.pf_cap += 1; assert k < _NG_PF_RING + _NG_PF_CAP, "ngpf: out of capture slots"
                    else:                                                  # eager: ring (host funcs of one stream run in order)
                        k = self.pf_k; self.pf_k = (k + 1) % _NG_PF_RING
                    rc = self.pf_lib.ngpf_launch(self.pf, self.stream.cuda_stream, k, ids.data_ptr(), ids.numel()); assert rc == 0, rc
                if self.hot is not None:
                    _gather_rows_hot[(ids.numel(),)](self.bases, ids, self.buf, self.part_rows, self.nrows, *self.hot, RB=self.rb,
                                                     BD=triton.next_power_of_2(self.rb))
                else:
                    _gather_rows[(ids.numel(),)](self.bases, ids, self.buf, self.part_rows, self.nrows, RB=self.rb,
                                                 BD=triton.next_power_of_2(self.rb))

        @eager_break_during_capture
        def _finalize(self, out):
            """join the side stream and copy into a graph-pool output (same pattern as vLLM's pinned-host PLE)."""
            torch.cuda.current_stream().wait_stream(self.stream)
            out.copy_(self.buf[: out.shape[0]].reshape(out.shape[0], -1))

        def finalize_prefetch(self, total_tokens):
            out = self.buf.new_empty((total_tokens, self.nh * self.rb))
            self._finalize(out)
            return out if self.out_dtype == torch.uint8 else out.view(self.out_dtype)

        def forward(self, input_):
            raise RuntimeError("disk-backed PLE rows require start_prefetch() / finalize_prefetch()")

    return TritPLEMethod, RowsPLEEmbedding


def _safetensors_rows(root, prefix):
    """memmaps of the checkpoint's own n-gram shards <prefix>.shard_<i>.weight (raw safetensors data, no copy) ->
    ([(first_row, memmap [rows, row_bytes] uint8)], row_bytes, torch dtype)."""
    import json, struct
    wm = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    tail = prefix[prefix.index("layers."):]
    stem = "model.language_model." + tail
    names = sorted((k for k in wm if k.startswith(stem + ".shard_") and k.endswith(".weight")),
                   key=lambda k: int(k[len(stem + ".shard_"): -len(".weight")]))
    assert names, f"no n-gram shards for {stem}"
    DT = {"F8_E4M3": (torch.float8_e4m3fn, 1), "BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "U8": (torch.uint8, 1)}
    parts, r0, rb, dt = [], 0, None, None
    for k in names:
        fn = os.path.join(root, wm[k])
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
        e = hdr[k]; dt, esz = DT[e["dtype"]]; rows, cols = e["shape"]; a, b = e["data_offsets"]
        rb = cols * esz; assert b - a == rows * rb
        parts.append((r0, np.memmap(fn, dtype=np.uint8, mode="r", offset=8 + n + a, shape=(rows, rb)))); r0 += rows
    return parts, rb, dt


def _patch_qwen4exp():
    import vllm.models.qwen4_exp.nvidia.ngram_embedding as ng
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod
    TritPLEMethod, TritRowsPLEEmbedding = _ple_classes()   # (RowsPLEEmbedding)
    orig = Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        if isinstance(quant_config, TritConfig) and quant_config.ngram:
            return TritPLEMethod(quant_config.ngram)
        if isinstance(quant_config, TritConfig):
            return orig(None, prefix, embedding_dtype)
        return orig(quant_config, prefix, embedding_dtype)
    Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(from_quant_config)

    def route(original):
        def make(*a, **kw):
            if isinstance(kw.get("embedding_method"), TritPLEMethod) or os.environ.get("TRIT_PLE_MMAP") == "1":
                return TritRowsPLEEmbedding(*a, **kw)
            return original(*a, **kw)
        make.supports_prefetch = True
        return make
    ng.Qwen4ExpPLEPinnedHostEmbedding = route(ng.Qwen4ExpPLEPinnedHostEmbedding)
    # Qwen4ExpForCausalLM builds ParallelLMHead WITHOUT quant_config -> the head could only load bf16. Rebuild it with ours.
    import vllm.models.qwen4_exp.nvidia.model as qm
    from vllm.model_executor.models.utils import maybe_prefix
    orig_init = qm.Qwen4ExpForCausalLM.__init__

    def init(self, *, vllm_config, prefix=""):
        orig_init(self, vllm_config=vllm_config, prefix=prefix)
        qc = vllm_config.quant_config
        if isinstance(qc, TritConfig):
            self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size, quant_config=qc,
                                          prefix=maybe_prefix(prefix, "lm_head"))
            self.lm_head.tq_is_head = True                   # draft vocabulary (the drafter shares this module, vLLM _maybe_share_lm_head)
    qm.Qwen4ExpForCausalLM.__init__ = init
    # the MTP drafter builds its own ParallelLMHead the same way (it loads the shared lm_head.* tensors) -> same fix
    import vllm.models.qwen4_exp.nvidia.mtp as mm
    orig_mtp_init = mm.Qwen4ExpMTP.__init__

    def mtp_init(self, *, vllm_config, prefix=""):
        orig_mtp_init(self, vllm_config=vllm_config, prefix=prefix)
        qc = vllm_config.quant_config
        if isinstance(qc, TritConfig) and isinstance(getattr(self, "lm_head", None), ParallelLMHead):
            cfg = vllm_config.model_config.hf_text_config
            self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size, quant_config=qc, prefix=maybe_prefix(prefix, "lm_head"))
            self.lm_head.tq_is_head = True
    mm.Qwen4ExpMTP.__init__ = mtp_init
    if _DRAFT_N:                                             # draft vocabulary: only the drafter's logits use the vocab prefix
        orig_cl = mm.Qwen4ExpMTP.compute_logits

        def mtp_compute_logits(self, hidden_states, spec_step_idx=0):
            _IN_DRAFT[0] = True
            try:
                return orig_cl(self, hidden_states, spec_step_idx)
            finally:
                _IN_DRAFT[0] = False
        mm.Qwen4ExpMTP.compute_logits = mtp_compute_logits
    ng.Qwen4ExpPLEDeviceEmbedding = route(ng.Qwen4ExpPLEDeviceEmbedding)


def _patch_gates():
    """TQ_L3: the MoE router (gate) and shared_expert_gate are built with quant_config=None; swap their linear method after construction
    (Qwen4ExpSparseMoeBlock subclasses Qwen3NextSparseMoeBlock). enable_qwen4_exp_low_latency_gemm (end of model init) only replaces
    methods whose type IS UnquantizedLinearMethod, so these stay on the Triton GEMV. Target model only (prefix without 'mtp')."""
    import vllm.model_executor.models.qwen3_next as qn
    orig = qn.Qwen3NextSparseMoeBlock.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        pfx = kw.get("prefix", a[1] if len(a) > 1 else "")
        if "mtp" in str(pfx):
            return
        for name in ("gate", "shared_expert_gate"):
            m = getattr(self, name, None)
            if m is not None and type(m.quant_method) is UnquantizedLinearMethod:
                m.quant_method = Bf16GemvMethod()
    qn.Qwen3NextSparseMoeBlock.__init__ = init


def _patch_hc8():
    """TQ_HC8: the HC projections (nvidia GatedResidual: merged down+inject [lora + hc + pad, HC*H], up [HC*H, lora], final mixer down)
    are built with quant_config=None; swap their linear method after construction and turn off the fused bf16 down+SiLU CuTe kernel so
    the down / inject projection goes through the module (GEMM -> split -> hc_silu, vLLM's own unfused path). Target model only."""
    import vllm.models.qwen4_exp.nvidia.hyperconnection as hcm
    orig = hcm.GatedResidual.__init__

    def init(self, config, use_combine=True, prefix=""):
        orig(self, config, use_combine=use_combine, prefix=prefix)
        if "mtp" in prefix:
            return
        if use_combine:
            self.input_mix_weight_down_block_inject.quant_method = HC8Method(self.lora_rank + self.hc_count)
            self._use_hc_down_silu = False
        else:
            self.input_mix_weight_down.quant_method = HC8Method()
        self.input_mix_weight_up.quant_method = HC8Method()
        self._tq_hc8 = True
    hcm.GatedResidual.__init__ = init
    orig_dai = hcm.GatedResidual._down_and_inject

    def down_and_inject(self, xn):
        """HC8 + TQ_HC8_FUSE: one int8 GEMM with vLLM's hc_silu fused on the lora columns (= GEMM -> split -> hc_silu, bitwise,
        checked by a unit test)"""
        if not getattr(self, "_tq_hc8", False) or not _HC8_FUSE:
            return orig_dai(self, xn)
        m = self.input_mix_weight_down_block_inject if self.use_combine else self.input_mix_weight_down
        y = m.quant_method.apply(m, xn, silu_n=self.lora_rank, hc=self.hc_count)
        if not self.use_combine:
            return y, None
        return y[..., : self.lora_rank], y[..., self.lora_rank: self.lora_rank + self.hc_count]
    hcm.GatedResidual._down_and_inject = down_and_inject



def _patch_skinny_fill():
    """TQ_MX_SKINNY: fill vLLM 0.32's skinny BF16 GEMM plans so that a 3- / 5- / 6- / 7-row speculative verify does not fall back to cuBLAS.
    Derived from salient-data/qwen38-next-dgx-recipe @f309019, recipe/patches/0011-skinny-gemm-rows.patch (Copyright 2026 Salient Data,
    Apache-2.0): the same fill rule, here applied at run time to every capability table (the patch edits the SM121 table in place).
    Every missing row count m <= max planned gets the next larger plan's config with num_rows = m."""
    import dataclasses
    import vllm.models.qwen4_exp.nvidia.low_latency_gemm as llg
    n = 0
    for plans in llg.QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY.values():
        for plan in plans.values():
            have = sorted(plan)
            for m in range(1, have[-1] + 1):
                if m not in plan:
                    plan[m] = dataclasses.replace(plan[next(q for q in have if q > m)], num_rows=m); n += 1
    logger.warning("tritq: skinny GEMM plans filled (%d row counts added)", n)


def register():
    """vllm.general_plugins entry point: registers the 'trit' quantization config and patches Qwen4Exp's PLE table selection."""
    try:
        _patch_qwen4exp()
    except Exception as e:                                   # noqa: BLE001 - the plugin must not break other models
        logger.warning("trit: Qwen4Exp PLE patch not applied: %s", e)
    for on, fn in ((_L3, _patch_gates), (_HC8, _patch_hc8), (os.environ.get("TQ_MX_SKINNY") == "1", _patch_skinny_fill)):
        if on:
            try:
                fn()
            except Exception as e:                           # noqa: BLE001
                logger.warning("trit: %s not applied: %s", fn.__name__, e)
    logger.warning("trit glue levers: L1=%s L2=%s L3=%s HC8=%s DRAFT_VOCAB=%s", _L1, _L2, _L3, _HC8, _DRAFT_N)
    logger.warning("trit v2 kernels: K_DEC=%s K_MOE=%s K_HC=%s K_PDL=%s (tk2 PDL available: %s)", _K_DEC, _K_MOE, _K_HC, _K_PDL, tk2.HAVE_PDL)
    return None

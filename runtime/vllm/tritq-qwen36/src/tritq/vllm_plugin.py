# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""vLLM quantization method "trit": dense-trit weights TQ9_128 / TQ27_128 (fmt.py) run by the Triton kernels in kern.py.
config.json quantization_config = {"quant_method": "trit", "modules": {"<module key>": N}}; module key = "lm_head" or
"layers.<L>.<vLLM module>" with the vLLM (fused) module names: linear_attn.in_proj_qkvz, linear_attn.out_proj, self_attn.qkv_proj,
self_attn.o_proj, mlp.shared_expert.gate_up_proj, mlp.shared_expert.down_proj, mlp.experts.w13, mlp.experts.w2.
A module without a key (routers, in_proj_ba, vision tower) stays unquantized (bf16 in the checkpoint).
Per-part levels: a fused module's value may be a LIST, one level per vLLM output partition (qkv_proj [q,k,v],
in_proj_qkvz [q,k,v,z], gate_up_proj [gate,up]); tq_key / tq_lo span all columns, tq_hi only the TQ27 partitions' columns.
"fp8_modules": prefixes (e.g. "mtp.") whose linears / experts load vLLM's block-FP8 checkpoint format (weight + weight_scale_inv,
"fp8_block" [128, 128], dynamic activations) through vLLM's own Fp8 methods: the MTP head at 8 bits.
Checkpoint tensors per quantized matrix: <name>.tq_key [G, out] fp16, <name>.tq_lo [G, 26, out] int16, <name>.tq_hi [G, 26, out] uint8 (N=27).
Every loader narrows along the LAST (output) dim, so vLLM's own merged / QKV / per-expert loaders place the shards."""
import os
import re
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
from . import kern, glue

logger = init_logger(__name__)
G, NW = 128, 26


def _tq_linear_op(x: torch.Tensor, key: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, nlev: int, segs: list[int]) -> torch.Tensor:
    """segs: [] = one container nlev; else flat (col0, ncol, nlev, hcol0) * n for a fused module with per-part levels.
    TQ_L2=1: split-K reduced inside the kernel (glue.tq_linear_rk; bit-identical)."""
    lin = glue.tq_linear_rk if _L2 else kern.tq_linear
    if segs:
        sg = [tuple(segs[i: i + 4]) for i in range(0, len(segs), 4)]
        return lin(x, key, lo, hi if any(t[2] == 27 for t in sg) else None, 0, segs=sg)
    return lin(x, key, lo, hi if nlev == 27 else None, nlev)


def _tq_linear_fake(x, key, lo, hi, nlev, segs):
    return torch.empty(x.shape[0], key.shape[1], dtype=x.dtype, device=x.device)


direct_register_custom_op("tq_linear", _tq_linear_op, mutates_args=[], fake_impl=_tq_linear_fake)

# glue options, read once per process (env of the vLLM server / engine): TQ_L1 decode MoE path, TQ_L2 in-kernel split-K reduce,
# TQ_L3 Triton bf16 GEMV for the unquantized routers / shared_expert_gate / in_proj_ba (order-changing vs cuBLAS)
_L1, _L2, _L3 = glue.flag("TQ_L1"), glue.flag("TQ_L2"), glue.flag("TQ_L3")
# Draft vocabulary option (reduced-vocab MTP draft head). TQ_DRAFT_VOCAB unset / "0" = off (= the base plugin); "1" = the default
# 100000 rows; any other integer = that many rows. The drafter's lm_head = rows [0, V) of the target's TQ9 lm_head (a contiguous copy of the
# column prefix of tq_key / tq_lo / tq_hi, built once at load; no new weights); the drafter argmaxes over that prefix, so ids >= V cannot be
# drafted. Verification keeps the full head. Same tile config as the full head (_cfg: split 1 at both widths) -> the prefix logits are
# bit-identical to the full head's first V columns (unit gate).
_DV = int(os.environ.get("TQ_DRAFT_VOCAB", "0") or 0)
_DV = 100000 if _DV == 1 else _DV
# Sparse draft head option: TQ_SPARSE="F,r,J" = sparse context-aware draft head (sparse.py): prefix [0, F) + J rows per draft token picked by a
# rank-r projection, exact logits for both; replaces the draft-vocabulary prefix (sets _DV = F for the prefix copy). Unset = off.
_SP = tuple(int(v) for v in os.environ.get("TQ_SPARSE", "").split(",")) if os.environ.get("TQ_SPARSE") else None
if _SP: _DV = _SP[0]


def _gemv_op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return glue.bf16_gemv(x, w)


def _gemv_fake(x, w):
    return torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)


direct_register_custom_op("tq_gemv_bf16", _gemv_op, mutates_args=[], fake_impl=_gemv_fake)


class Bf16GemvMethod(UnquantizedLinearMethod):
    """bf16 weights as stored; decode-size inputs (M <= 8, no bias) go through one Triton GEMV instead of cuBLAS split-K + reduce"""
    def apply(self, layer, x, bias=None):
        x2 = x.reshape(-1, x.shape[-1])
        if bias is None and x2.shape[0] <= 8 and layer.weight.dtype == torch.bfloat16 and x2.dtype == torch.bfloat16:
            y = torch.ops.vllm.tq_gemv_bf16(x2.contiguous(), layer.weight)
            return y.reshape(*x.shape[:-1], y.shape[-1])
        return super().apply(layer, x, bias)


def _module_key(prefix: str):
    if "mtp" in prefix or "visual" in prefix:
        return None
    if prefix.endswith("lm_head"):
        return "lm_head"
    m = re.search(r"layers\.(\d+)\.(.+)$", prefix)
    return f"layers.{m.group(1)}.{m.group(2)}" if m else None


@register_quantization_config("trit")
class TritConfig(QuantizationConfig):
    def __init__(self, modules: dict, fp8_modules=None, fp8_block=None):
        super().__init__()
        self.modules = modules
        self.seen = set()
        self.fp8_modules = list(fp8_modules or [])
        self.fp8 = None
        if self.fp8_modules:
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config
            self.fp8 = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=fp8_block or [128, 128])

    def __repr__(self):
        return f"TritConfig({len(self.modules)} modules)"

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
        return cls(config["modules"], config.get("fp8_modules"), config.get("fp8_block"))

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        if self.fp8 is not None and any(prefix.startswith(m) or ("." + m) in ("." + prefix) for m in self.fp8_modules) \
                and not prefix.endswith((".gate", "shared_expert_gate")):
            if isinstance(layer, (LinearBase, RoutedExperts)):
                self.fp8.packed_modules_mapping = self.packed_modules_mapping
                return self.fp8.get_quant_method(layer, prefix)
        k = _module_key(prefix)
        if isinstance(layer, RoutedExperts):
            k13, k2 = (k + ".w13", k + ".w2") if k else (None, None)
            if k13 in self.modules:
                self.seen.add(k13); self.seen.add(k2)
                return TritMoEMethod(self.modules[k13], self.modules[k2], layer.moe_config)
            logger.info("trit: unquantized experts %s", prefix)
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
            return UnquantizedFusedMoEMethod(layer.moe_config)
        if isinstance(layer, ParallelLMHead) or isinstance(layer, LinearBase):
            if k in self.modules:
                self.seen.add(k)
                return TritLinearMethod(self.modules[k], head=(k == "lm_head"))
            if isinstance(layer, ParallelLMHead):
                from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
                return UnquantizedEmbeddingMethod()
            return Bf16GemvMethod() if _L3 else UnquantizedLinearMethod()
        return None


class TritLinearMethod(LinearMethodBase):
    def __init__(self, nlev, head=False):
        self.head = head
        self.levels = nlev if isinstance(nlev, list) else None        # per output partition, or None = one container
        self.nlev = max(nlev) if isinstance(nlev, list) else nlev
        if self.levels is not None and len(set(self.levels)) == 1:
            self.levels = None

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype,
                       **extra_weight_attrs):
        K = input_size_per_partition; NO = sum(output_partition_sizes); assert K % G == 0, (K, layer)
        ng = K // G; wl = extra_weight_attrs.get("weight_loader")
        key = Parameter(torch.empty(ng, NO, dtype=torch.float16), requires_grad=False)
        lo = Parameter(torch.empty(ng, NW, NO, dtype=torch.int16), requires_grad=False)
        set_weight_attrs(key, {"output_dim": 1, "weight_loader": wl}); set_weight_attrs(lo, {"output_dim": 2, "weight_loader": wl})
        layer.register_parameter("tq_key", key); layer.register_parameter("tq_lo", lo)
        layer.tq_segs = []
        if self.levels is None:
            if self.nlev == 27:
                hi = Parameter(torch.empty(ng, NW, NO, dtype=torch.uint8), requires_grad=False)
                set_weight_attrs(hi, {"output_dim": 2, "weight_loader": wl}); layer.register_parameter("tq_hi", hi)
        else:
            sizes = list(output_partition_sizes); assert len(sizes) == len(self.levels), (sizes, self.levels, layer)
            offs = [sum(sizes[:i]) for i in range(len(sizes))]
            hoff, h = [], 0
            for sz, lv in zip(sizes, self.levels):
                hoff.append(h); h += sz if lv == 27 else 0
            segs = []                                                        # merge adjacent partitions of equal level
            for o, sz, lv, ho in zip(offs, sizes, self.levels, hoff):
                if segs and segs[-1][2] == lv: segs[-1][1] += sz
                else: segs.append([o, sz, lv, ho])
            layer.tq_segs = [v for t in segs for v in t]
            if h:
                hi = Parameter(torch.empty(ng, NW, h, dtype=torch.uint8), requires_grad=False)
                set_weight_attrs(hi, {"weight_loader": self._hi_loader(layer, sizes, hoff, self.levels)})
                layer.register_parameter("tq_hi", hi)
        layer.tq_nlev = self.nlev

    @staticmethod
    def _hi_loader(layer, sizes, hoff, levels):
        """tq_hi holds only the TQ27 partitions: place a shard (int / 'q','k','v' / tuple of partitions) at its offset in hi"""
        def load(param, loaded_weight, shard_id=None):
            if isinstance(shard_id, str): idx = ["q", "k", "v"].index(shard_id)
            elif isinstance(shard_id, tuple): idx = shard_id[0]
            elif shard_id is None: idx = 0
            else: idx = shard_id
            assert levels[idx] == 27, (layer, shard_id, levels)
            n = loaded_weight.shape[-1]
            dst = param.data.narrow(2, hoff[idx], n)
            assert dst.shape == loaded_weight.shape, (dst.shape, loaded_weight.shape)
            dst.copy_(loaded_weight)
        return load

    def process_weights_after_loading(self, layer):
        if not hasattr(layer, "tq_hi"):
            layer.tq_hi_t = layer.tq_lo                          # unused placeholder argument for the op
        if self.head and _DV > 0 and self.levels is None and _DV < layer.tq_key.shape[1]:
            V = _DV                                              # draft vocabulary: contiguous column prefix [0, V) for the MTP drafter
            hi = layer.tq_hi[:, :, :V].contiguous() if hasattr(layer, "tq_hi") else None
            layer.tq_dv = (layer.tq_key[:, :V].contiguous(), layer.tq_lo[:, :, :V].contiguous(), hi)
            print("[tritq] draft-vocab: draft head = rows [0, %d) of the lm_head (%d rows), %.1f MB" % (V, layer.tq_key.shape[1],
                  sum(t.numel() * t.element_size() for t in layer.tq_dv if t is not None) / 1e6), flush=True)
            if _SP:
                from .sparse import SparseHead
                layer.tq_sparse = SparseHead(layer.tq_key, layer.tq_lo, getattr(layer, "tq_hi", None), layer.tq_nlev, *_SP)
                print("[tritq] sparse draft head F,r,J(,sc)=%s, %.1f MB extra (scorer %.1f MB)" % (_SP, layer.tq_sparse.mb, layer.tq_sparse.scorer_mb), flush=True)

    def apply(self, layer, x, bias=None):
        shp = x.shape; x2 = x.reshape(-1, shp[-1]).contiguous()
        hi = layer.tq_hi if hasattr(layer, "tq_hi") else layer.tq_hi_t
        y = torch.ops.vllm.tq_linear(x2, layer.tq_key, layer.tq_lo, hi, self.nlev, layer.tq_segs)
        if bias is not None: y = y + bias
        return y.reshape(*shp[:-1], y.shape[-1])

    # ParallelLMHead calls quant_method.embedding() only for tied embeddings; not used here


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
        if _L1 and x.shape[0] * topk_ids.shape[1] <= 64 and isinstance(self.n13, int) and isinstance(self.n2, int):
            return glue.tq_moe_decode(x.contiguous(), w13, w2, topk_weights, topk_ids.to(torch.int32))     # bit-identical (unit gates)
        return kern.tq_moe(x.contiguous(), w13, w2, topk_weights, topk_ids.to(torch.int32), layer.global_num_experts)


def _patch_gates():
    """TQ_L3: the MoE router (gate) and shared_expert_gate are built with quant_config=None, so the plugin never sees them; swap their
    linear method after construction."""
    import vllm.model_executor.models.qwen3_next as qn
    orig = qn.Qwen3NextSparseMoeBlock.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        for name in ("gate", "shared_expert_gate"):
            m = getattr(self, name, None)
            if m is not None and type(m.quant_method) is UnquantizedLinearMethod:
                m.quant_method = Bf16GemvMethod()
    qn.Qwen3NextSparseMoeBlock.__init__ = init


def draft_logits(lm_head, hidden_states):
    """draft vocabulary: the MTP drafter's logits over the prefix [0, V) of the target lm_head (None if the option is off)"""
    dv = getattr(lm_head, "tq_dv", None)
    if dv is None:
        return None
    key, lo, hi = dv
    shp = hidden_states.shape; x2 = hidden_states.reshape(-1, shp[-1]).contiguous()
    nlev = lm_head.tq_nlev
    y = torch.ops.vllm.tq_linear(x2, key, lo, hi if hi is not None else lo, nlev, [])
    sh = getattr(lm_head, "tq_sparse", None)
    if sh is not None:
        y = sh.logits(x2, y)
    return y.reshape(*shp[:-1], y.shape[-1])


def _patch_mtp_head():
    """draft vocabulary: Qwen3_5MTP.compute_logits (the drafter; vLLM 0.28 EagleProposer._greedy_sample = compute_logits(h).argmax) uses the prefix head.
    Its lm_head is the target's (shared by _maybe_share_lm_head). Only the greedy draft path is supported: draft_sample_method must be
    'greedy' (the default), since probabilistic draft probs would be V wide, not vocab wide."""
    import vllm.model_executor.models.qwen3_5_mtp as qm
    orig = qm.Qwen3_5MTP.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx=0):
        y = draft_logits(self.lm_head, hidden_states)
        return orig(self, hidden_states, spec_step_idx) if y is None else y
    qm.Qwen3_5MTP.compute_logits = compute_logits


def register():
    """vllm.general_plugins entry point: importing this module registers the 'trit' quantization config."""
    if _L3:
        try:
            _patch_gates()
        except Exception as e:                                    # noqa: BLE001
            logger.warning("trit: gate patch not applied: %s", e)
    if _DV > 0:
        _patch_mtp_head()
    print("[tritq] options L1=%s L2=%s L3=%s draft_vocab=%s sparse=%s" % (_L1, _L2, _L3, _DV or "off", _SP), flush=True)
    return None

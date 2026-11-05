# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""tritq125: vLLM plugin that adds the "trit" quantization method for Qwen3.8-Flash-Next TR models (packed k-trit weights, hot/cold
expert sets, a packed-trit n-gram (PLE) table read from disk, Triton kernels).

Runtime options are environment variables, read once per process when vLLM loads the plugin. The public names (TRITQ_*) map onto the
names the kernels read (TQ_* / TRIT_*); setting either works. Defaults below are the release configuration.

  TRITQ_FUSED_MOE       (TQ_L1)          1  decode-time MoE path without the align/sort step (bit-identical)
  TRITQ_SPLITK_REDUCE   (TQ_L2)          1  split-K partial sums reduced inside the kernel (bit-identical)
  TRITQ_BF16_GEMV       (TQ_L3)          0  Triton GEMV for small unquantized linears (order-changing; not used for results)
  TRITQ_HC_INT8         (TQ_HC8)         1  hyper-connection projections at 8 bits (Q8_0 rule, quantized at load; value-changing,
                                            measured no significant accuracy change)
  TRITQ_SKINNY_FILL     (TQ_MX_SKINNY)   1  fill vLLM's skinny BF16 GEMM plans for every row count up to the largest planned one, so
                                            3- to 7-row speculative verifies of the BF16 routers / gates do not fall back to cuBLAS
  TRITQ_NGRAM_PREFETCH  (TQ_NG_PF)       1  read the n-gram table pages a step needs in parallel (CPU host function in stream order;
                                            reads only, never changes the gathered bytes). Needs gcc at first start (builds ngpf.c).
  TRITQ_KERNELS_V2      (TQ_K_DEC, TQ_K_MOE, TQ_K_HC)  1  v2 kernels (tk2.py): per-shape tiles for the dense trit linears, the batched
                                            routed-expert path for 64 < slots and T <= 32 rows, the 8-bit HC GEMM at > 16 rows. Bit-identical
                                            to the kernels they replace (greedy text identical at one request); each TQ_K_* can be set to 0.
  TRITQ_MTP_FP8         (TQ_MTP_FP8)     1  the MTP draft head's linears and routed experts quantised to FP8 (per tensor) at load. Changes
                                            only the drafts, but under the confidence stop that changes how many drafts are verified, so the
                                            served text changes (value-changing; measured no significant accuracy change). 0 = BF16 draft head.
  TRITQ_DRAFT_HEAD                       sparse  which head the MTP drafter uses (verification always uses the full head; only the drafts,
                                            i.e. the speed, and through the confidence stop the served text, depend on this choice):
        sparse   all-language context-aware sparse head (default): an exact prefix of 32,768 rows + 16,384 rows per draft token picked
                 by a rank-256 projection of the head (TQ_SPARSE="32768,256,16384,1"); keeps drafting fast in every language.
        english  reduced-vocabulary draft head over the first 100,000 rows. Fastest on English and code; slower than the sparse head
                 on Chinese, Japanese, Russian and Korean.
        full     the full-vocabulary draft head.
  TRITQ_DRAFT_VOCAB     (TQ_DRAFT_VOCAB)    explicit reduced-vocabulary size (1 = 100000; 0 = full head).
  TRITQ_PLE_MMAP        (TRIT_PLE_MMAP)  0  route another checkpoint's FP8 n-gram rows through the same disk-backed lookup
                                            (for benchmarking other checkpoints; not needed for TR checkpoints)
Diagnostics (off): TQ_NG_PFSTAT=1 counts non-resident pages; TQ_NG_DUMP=<file> logs looked-up ids; TQ_NG_RANDOM=1 MADV_RANDOM;
TQ_NG_HOT=<dir> resident hot-row set (experimental).

The confidence-stop draft depth is an engine option, not a plugin option: the Docker image applies vLLM PR #60068 (patches/), and the
server takes --speculative-config '{"method":"mtp","num_speculative_tokens":6,"draft_confidence_threshold":0.7,
"draft_confidence_fallback_depth":3}'.
"""
import os

_ALIASES = {"TRITQ_FUSED_MOE": "TQ_L1", "TRITQ_SPLITK_REDUCE": "TQ_L2", "TRITQ_BF16_GEMV": "TQ_L3", "TRITQ_HC_INT8": "TQ_HC8",
            "TRITQ_SKINNY_FILL": "TQ_MX_SKINNY", "TRITQ_NGRAM_PREFETCH": "TQ_NG_PF", "TRITQ_DRAFT_VOCAB": "TQ_DRAFT_VOCAB",
            "TRITQ_PLE_MMAP": "TRIT_PLE_MMAP", "TRITQ_MTP_FP8": "TQ_MTP_FP8"}
_DEFAULTS = {"TQ_L1": "1", "TQ_L2": "1", "TQ_HC8": "1", "TQ_MX_SKINNY": "1", "TQ_NG_PF": "1", "TQ_MTP_FP8": "1"}
_SPARSE_DEFAULT = "32768,256,16384,1"
_SPARSE_VOC = "248077"                               # real vocabulary of Qwen3.8-Flash-Next (largest token id 248,076)

for _pub, _int in _ALIASES.items():
    if _pub in os.environ and _int not in os.environ:
        os.environ[_int] = os.environ[_pub]
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)
if "TRITQ_KERNELS_V2" in os.environ:                 # one public switch for the three v2 kernel flags
    for _k in ("TQ_K_DEC", "TQ_K_MOE", "TQ_K_HC"):
        os.environ.setdefault(_k, os.environ["TRITQ_KERNELS_V2"])
for _k in ("TQ_K_DEC", "TQ_K_MOE", "TQ_K_HC"):
    os.environ.setdefault(_k, "1")

_head = os.environ.get("TRITQ_DRAFT_HEAD", "").strip().lower()
if not _head:     # no explicit choice: an explicit TQ_SPARSE / TQ_DRAFT_VOCAB wins, otherwise the sparse head
    _head = "custom" if ("TQ_SPARSE" in os.environ or "TQ_DRAFT_VOCAB" in os.environ) else "sparse"
if _head == "sparse":
    os.environ.setdefault("TQ_SPARSE", _SPARSE_DEFAULT)
    os.environ.pop("TQ_DRAFT_VOCAB", None)
elif _head == "english":
    os.environ.pop("TQ_SPARSE", None)
    os.environ["TQ_DRAFT_VOCAB"] = "100000"
elif _head == "full":
    os.environ.pop("TQ_SPARSE", None)
    os.environ["TQ_DRAFT_VOCAB"] = "0"
elif _head != "custom":
    raise ValueError(f"TRITQ_DRAFT_HEAD={_head!r}: expected sparse, english or full")
if "TQ_SPARSE" in os.environ:
    os.environ.setdefault("TQ_SPARSE_VOC", _SPARSE_VOC)
if os.environ.get("TQ_DRAFT_VOCAB") == "1":          # same meaning as in the 35B plugin: 1 = the first 100000 rows
    os.environ["TQ_DRAFT_VOCAB"] = "100000"

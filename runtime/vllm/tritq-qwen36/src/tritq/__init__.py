# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""tritq: vLLM plugin that adds the "trit" quantization method (packed k-trit weights, Triton kernels) for Qwen3.6-35B-A3B TR models.

Runtime options are environment variables, read once per process when vLLM loads the plugin. The public names (TRITQ_*) map onto the
names the kernels read (TQ_*); setting either works. Defaults below are the release configuration.

  TRITQ_FUSED_MOE      (TQ_L1)          1  decode-time MoE path without the align/sort step (bit-identical to the reference path)
  TRITQ_SPLITK_REDUCE  (TQ_L2)          1  split-K partial sums reduced inside the kernel (bit-identical)
  TRITQ_BF16_GEMV      (TQ_L3)          0  Triton GEMV for the small unquantized linears (changes summation order; not used for results)
  TRITQ_DRAFT_HEAD                      sparse  which head the MTP drafter uses (verification always uses the full head, so greedy output
                                                text does not depend on this choice; only the draft acceptance, i.e. the speed, does):
        sparse   all-language sparse draft head (default): exact logits for the first 32,768 vocabulary rows plus 8,192 rows per draft
                 token chosen by a rank-256 projection of the head (TQ_SPARSE="32768,256,8192,1"). Keeps drafting fast in every
                 language; costs ~0.44 GiB of GPU memory.
        english  reduced-vocabulary draft head over the first 100,000 rows (TQ_DRAFT_VOCAB=100000). Fastest on English and code,
                 measurably slower on Chinese, Japanese, Russian and Korean. Use only for English/code traffic.
        full     the full-vocabulary draft head.
  TRITQ_DRAFT_VOCAB    (TQ_DRAFT_VOCAB)    explicit reduced-vocabulary size (1 = 100000). Setting it (or TQ_SPARSE) without
                                           TRITQ_DRAFT_HEAD selects that head instead of the sparse default.

Engine options that are not plugin options (set by the Docker image, see docker/Dockerfile.qwen36-vllm028):
  VLLM_USE_V2_MODEL_RUNNER=1   vLLM's Model Runner V2 (the release configuration; needs this plugin's expert-id-masked MoE path).
  TQ_CSTOP="0.7,3"             confidence-stop draft depth (vllm028_overlay/, a port of vLLM PR #60068): with one request decoding,
                               stop drafting once the drafter's top-1 probability is below 0.7 (up to num_speculative_tokens, use 6);
                               with two or more requests, draft 3. Set TQ_CSTOP= (empty) and num_speculative_tokens 3 for
                               multi-user serving, where the stop costs ~2 % per stream.
"""
import os

_ALIASES = {"TRITQ_FUSED_MOE": "TQ_L1", "TRITQ_SPLITK_REDUCE": "TQ_L2", "TRITQ_BF16_GEMV": "TQ_L3", "TRITQ_DRAFT_VOCAB": "TQ_DRAFT_VOCAB"}
_DEFAULTS = {"TQ_L1": "1", "TQ_L2": "1"}
_SPARSE_DEFAULT = "32768,256,8192,1"     # F (fixed prefix), r (projection rank), J (rows per draft token), scorer stored as TQ27

for _pub, _int in _ALIASES.items():
    if _pub in os.environ and _int not in os.environ:
        os.environ[_int] = os.environ[_pub]
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

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
    os.environ.pop("TQ_DRAFT_VOCAB", None)
elif _head != "custom":
    raise ValueError(f"TRITQ_DRAFT_HEAD={_head!r}: expected sparse, english or full")

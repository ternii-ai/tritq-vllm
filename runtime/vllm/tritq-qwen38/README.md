# tritq-qwen38: TR Quantisation plugin for Qwen3.8-Flash-Next TR models (vLLM 0.32.0)

Adds the quantization method `"trit"` to vLLM 0.32.0 for Qwen3.8-Flash-Next (`Qwen4Exp`): packed k-trit weights, two-depth
(hot/cold) routed-expert sets, the n-gram (PLE) table read from a packed file through the page cache (with a parallel page
read-ahead), 8-bit hyper-connections, a skinny-GEMM plan fill for short speculative verifies, v2 kernels (`tk2.py`), an FP8 MTP draft head and the
all-language sparse draft head (`sparse.py`). `patches/` holds vLLM PR #60068 (confidence-stop draft depth), applied by the Docker image.

**Defaults since 2026-10-27 (release configuration), each with its off-switch:**

| option | default | off-switch | effect on output |
|---|---|---|---|
| v2 kernels (per-shape dense tiles, batched routed-expert path, 8-bit HC GEMM at > 16 rows) | on | `TRITQ_KERNELS_V2=0` (or a single `TQ_K_DEC` / `TQ_K_MOE` / `TQ_K_HC=0`) | greedy text identical at one request (multi-user not checked) |
| FP8 MTP draft head (linears + routed experts, per tensor, quantised at load) | on | `TRITQ_MTP_FP8=0` (BF16 draft head) | value-changing through the confidence stop |
| draft head | `sparse` (all languages) | `TRITQ_DRAFT_HEAD=english` (first 100,000 rows: fastest on English / code) or `full` | value-changing through the confidence stop |

- Verification always uses the full model and the full vocabulary, but the drafts set how many tokens are verified per step under the
  confidence stop, and the target kernels are width-dependent, so **the served text differs from the previous release configuration
  (English-and-code head, BF16 draft head, no v2 kernels) on ~15 of 40 benchmark prompts**. Measured accuracy (GSM8K / HumanEval / MBPP)
  shows no significant difference (not an equivalence test).
- Startup takes about 11–12 minutes per start (kernel compilation) unless a cache directory is mounted; with no `--host` the
  server listens on 0.0.0.0:8000. vLLM's "Using default MoE config" warning for the FP8 draft experts is expected.
- Weights as loaded: **63.84 GiB** (was 65.66 GiB: −2.41 GiB from the FP8 draft head, +0.59 GiB for the sparse head's tables).
- With the FP8 draft head, `--moe-backend triton` selects vLLM's Triton FP8 MoE backend for the draft head's routed experts.
- `TQ_K_DEC` needs the in-kernel split-K reduce: `TRITQ_SPLITK_REDUCE=0` silently disables it. An empty `TQ_SPARSE=` falls through to
  the full draft head.
- Use the Docker image tag `tritq-qwen38:0.3`; a `tritq-qwen38:0.2` image carries the old source.

See `../../README.md` for the quick start and `src/tritq125/__init__.py` for the options.

Tested: vLLM 0.32.0 on one NVIDIA DGX Spark (GB10, sm_121), CUDA 13.0. Needs `gcc` in the image for the n-gram read-ahead (built at
first start). Other GPUs: not tested.

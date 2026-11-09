# TR Quantisation runtimes: quick start

TR models are Qwen models whose weights are stored as packed ternary-digit (trit) codes with one 16-bit scale per group ("TR
Quantisation"). They need a small runtime addition:

| runtime | for | contents |
|---|---|---|
| `vllm/tritq-qwen36` | Qwen3.6-35B-A3B TR checkpoints on **vLLM 0.28.0** | pip package (vLLM plugin entry point `tritq`) + `vllm028_overlay/` (confidence-stop draft depth, 7 vLLM 0.28.0 files) |
| `vllm/tritq-qwen38` | Qwen3.8-Flash-Next TR checkpoints on **vLLM 0.32.0** | pip package (entry point `tritq125`) + `patches/` (vLLM PR #60068, applied by the Dockerfile) |
| `docker/` | one image per model family (DGX Spark, aarch64, CUDA 13) | Dockerfiles; base images are not fetched here |
| `llama.cpp/` | GGUF files on Macs (Metal); aarch64 CPU kernels (patches 0005-0006); tools/ for trained ternary models | patch series on ggml-org/llama.cpp @ff5a233 + build notes |
| `bench/` | speed measurement | our OpenAI-API client + a pinned wrapper for the community harness |

**Version split.** The Qwen3.6-35B-A3B runtime was measured on vLLM 0.28.0; the Qwen3.8-Flash-Next runtime needs vLLM 0.32.0 (the first
release with Qwen3.8-Flash-Next).
The two plugins both register the quantization method `"trit"`, so install one per environment.

**Draft heads (MTP speculative decoding).** Verification always uses the full model and the full vocabulary, and the draft head is
chosen for speed. It is not output-neutral, though: under the confidence stop the drafts decide how many tokens are verified per step,
and the target kernels are width-dependent, so the choice of draft head (and, on Qwen3.8-Flash-Next, the FP8 draft
head) changes the served text on some prompts. Measured accuracy shows no significant difference (not an equivalence test).

| model | default draft head | option |
|---|---|---|
| Qwen3.6-35B-A3B | **sparse, all languages** (`TRITQ_DRAFT_HEAD=sparse`) | `english` (first 100,000 vocabulary rows: fastest on English and code, slower on other languages), `full` |
| Qwen3.8-Flash-Next | **sparse, all languages** (`TRITQ_DRAFT_HEAD=sparse`) | `english` (first 100,000 vocabulary rows: fastest on English and code, slower on other languages), `full` |

## DGX Spark, Qwen3.6-35B-A3B TR2.25-G (vLLM 0.28.0)

```bash
docker pull ghcr.io/ternii-ai/tritq-qwen36:0.2 && docker tag ghcr.io/ternii-ai/tritq-qwen36:0.2 tritq-qwen36:0.2
# or build it yourself from this folder: docker build -f docker/Dockerfile.qwen36-vllm028 -t tritq-qwen36:0.2 .
docker run --rm --gpus all --ipc=host --network host --ulimit memlock=-1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  tritq-qwen36:0.2 Ternii/Qwen3.6-35B-A3B-TR2.25-G \
  --served-model-name tr35 --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.6 --kv-cache-dtype fp8 \
  --language-model-only --speculative-config '{"method":"mtp","num_speculative_tokens":6}'
```

- The image sets `VLLM_USE_V2_MODEL_RUNNER=1` (Model Runner V2) and `TQ_CSTOP=0.7,3` (confidence stop: with one request decoding,
  draft up to `num_speculative_tokens` = 6 and stop once the drafter's top-1 probability is below 0.7; with two or more requests,
  draft 3).
- **Several users:** the confidence stop costs about 2 % per stream at 4-8 users. For multi-user serving run with `-e TQ_CSTOP=` and
  `"num_speculative_tokens":3` (plain Model Runner V2).
- Draft head: `-e TRITQ_DRAFT_HEAD=english` for English/code-only traffic; `full` for the full-vocabulary head.

## DGX Spark, Qwen3.8-Flash-Next TR2.57-G (vLLM 0.32.0)

```bash
docker pull ghcr.io/ternii-ai/tritq-qwen38:0.3 && docker tag ghcr.io/ternii-ai/tritq-qwen38:0.3 tritq-qwen38:0.3
# or build it yourself from this folder: docker build -f docker/Dockerfile.qwen38-vllm032 -t tritq-qwen38:0.3 .
docker run --rm --gpus all --ipc=host --network host --ulimit memlock=-1 --shm-size 16g \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  tritq-qwen38:0.3 Ternii/Qwen3.8-Flash-Next-TR2.57-G \
  --served-model-name qwen38-tr --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 10737418240 --language-model-only --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":6,"draft_confidence_threshold":0.7,"draft_confidence_fallback_depth":3}'
```

- Plugin defaults: fused MoE, in-kernel split-K reduce, 8-bit hyper-connections, skinny-GEMM plan fill, n-gram read-ahead, v2 kernels,
  FP8 MTP draft head and the all-language sparse draft head. Off-switches: `-e TRITQ_KERNELS_V2=0`,
  `-e TRITQ_MTP_FP8=0` (BF16 draft head), `-e TRITQ_DRAFT_HEAD=english` or `full`, `-e TRITQ_NGRAM_PREFETCH=0` (read-ahead).
- Compared with all three off (English-and-code head, BF16 draft head, no v2 kernels), the served text differs on
  ~15 of 40 benchmark prompts (the FP8 draft head changes the verify widths under the confidence stop; the v2 kernels alone keep the text
  at one request). Measured accuracy shows no significant difference (not an equivalence test). Weights as loaded: 63.84 GiB
  (66.24 GiB with the BF16 draft head, measured).
- The confidence stop comes from vLLM PR #60068 (applied in the image). Without `draft_confidence_threshold` it is inert; for plain
  fixed-depth drafting use `'{"method":"mtp","num_speculative_tokens":3}'`.
- The n-gram table (`ngram_tq27.bin`, ~34 GB) stays on disk and is read through the page cache; keep the model on a local NVMe drive.
- `--moe-backend triton` affects only the MTP draft head's routed experts: with the default FP8 draft head it selects vLLM's Triton FP8
  MoE backend (with `TRITQ_MTP_FP8=0`, the BF16 Triton path); otherwise vLLM picks a FlashInfer path whose JIT build needs `nvrtc.h`.
  The trit experts always use the plugin's kernels.
- `--kv-cache-memory-bytes 10737418240` (10 GiB) left at least 30 GB of host memory free in our runs on a 128 GB Spark. Raise it only if
  nothing else runs on the machine.
- The first start takes about 11-12 minutes (kernel compilation) unless a cache directory is mounted. Tested on one DGX Spark only.
- The `tritq-qwen38:0.3` image was built from the Dockerfile (the speed rows used an earlier hand-assembled image with the same
  recipe). Comments and docstrings in the plugin source have been edited since, so a rebuild is not byte-identical to it. Only the packages it names are pinned, and it has not yet been rebuilt on a clean machine.

Client side, both models: standard OpenAI-compatible API. Thinking is on by default in the Qwen chat template; pass
`"chat_template_kwargs": {"enable_thinking": false}` to turn it off (our benchmarks ran with thinking off).

## Without Docker

`pip install --no-deps` the plugin into a venv that already has the right vLLM. The 35B confidence stop additionally needs the
`vllm028_overlay/vllm/` files copied over the venv's vLLM 0.28.0 (check `SHA256SUMS.vllm-0.28.0-originals` first); without them, leave
`TQ_CSTOP` unset and use `num_speculative_tokens` 3. The Qwen3.8-Flash-Next confidence stop needs
`vllm/tritq-qwen38/patches/0010-…patch` applied to vLLM 0.32.0 (`patch -p1 --fuzz=0` in site-packages); without it, use
`'{"method":"mtp","num_speculative_tokens":3}'`.

## Mac (Apple silicon), GGUF

See `llama.cpp/README.md`: build the patched llama.cpp, then `llama-server -m <TR GGUF>`. For machines whose GPU budget cannot hold
the model, `LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB=<n>` keeps the experts on the SSD behind a fixed cache (experimental, measured at 4k
context only).

- Qwen3.6-35B-A3B: `Ternii/Qwen3.6-35B-A3B-TR2.25-G-GGUF`.
- Qwen3.8-Flash-Next: `Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF`, **experimental** and a different file from the vLLM checkpoint (linear
  layers and routed experts at 3 trits). It was timed only with the expert cache (four sizes) and is slow below 64 GB; its speeds were measured on
  one Apple M5 Max with memory capped, not on real smaller Macs. Cache sizes and the switches for its two decode fusions (patch 0003) are
  on its model card.

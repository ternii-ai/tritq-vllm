# TR Quantisation runtimes: quick start

TR models are Qwen models whose weights are stored as packed ternary-digit (trit) codes with one 16-bit scale per group ("TR
Quantisation"). They need a small runtime addition:

| runtime | for | contents |
|---|---|---|
| `vllm/tritq-qwen36` | Qwen3.6-35B-A3B TR checkpoints on **vLLM 0.28.0** | pip package (vLLM plugin entry point `tritq`) + `vllm028_overlay/` (confidence-stop draft depth, 7 vLLM 0.28.0 files) |
| `docker/` | the image for DGX Spark (aarch64, CUDA 13) | Dockerfile; base image is not fetched here |
| `llama.cpp/` | GGUF files on Macs (Metal) | patch series on ggml-org/llama.cpp @ff5a233 + build notes |
| `bench/` | speed measurement | our OpenAI-API client + a pinned wrapper for the community harness |

**Draft heads (MTP speculative decoding).** Verification always uses the full model and the full vocabulary, and the draft head is
chosen for speed. It is not output-neutral, though: under the confidence stop the drafts decide how many tokens are verified per step,
and the target kernels are width-dependent, so the choice of draft head changes the served text on
some prompts. Measured accuracy shows no significant difference (not an equivalence test).

| model | default draft head | option |
|---|---|---|
| Qwen3.6-35B-A3B | **sparse, all languages** (`TRITQ_DRAFT_HEAD=sparse`) | `english` (first 100,000 vocabulary rows: fastest on English and code, slower on other languages), `full` |

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

Client side: standard OpenAI-compatible API. Thinking is on by default in the Qwen chat template; pass
`"chat_template_kwargs": {"enable_thinking": false}` to turn it off (our benchmarks ran with thinking off).

## Without Docker

`pip install --no-deps` the plugin into a venv that already has the right vLLM. The 35B confidence stop additionally needs the
`vllm028_overlay/vllm/` files copied over the venv's vLLM 0.28.0 (check `SHA256SUMS.vllm-0.28.0-originals` first); without them, leave
`TQ_CSTOP` unset and use `num_speculative_tokens` 3.

## Mac (Apple silicon), GGUF

See `llama.cpp/README.md`: build the patched llama.cpp, then `llama-server -m <TR GGUF>`. For machines whose GPU budget cannot hold
the model, `LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB=<n>` keeps the experts on the SSD behind a fixed cache (experimental, measured at 4k
context only).

# tritq-vllm: TR Quantisation runtimes

TR models are Qwen models whose weights are stored as packed ternary-digit (trit) codes with one 16-bit scale per group of 128
weights ("TR Quantisation"). Stock vLLM and llama.cpp cannot read this format; this repository holds the small runtime additions
that can:

- `runtime/vllm/`: the vLLM plugin for DGX Spark (Qwen3.6-35B-A3B on vLLM 0.28.0)
- `runtime/docker/`: the Dockerfile for the image
- `runtime/llama.cpp/`: a patch series for llama.cpp on Apple silicon (Metal)
- `runtime/bench/`: the speed benchmarks used for the published figures

Start with [`runtime/README.md`](runtime/README.md).

## Qwen3.6-35B-A3B TR2.25-G vs NVIDIA's NVFP4 checkpoint

One DGX Spark, vLLM 0.28.0, same server settings, text only, thinking off; NVIDIA's `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`
unmodified on stock vLLM. Full results, methods and caveats are on the
[model card](https://huggingface.co/Ternii/Qwen3.6-35B-A3B-TR2.25-G).

| | TR2.25-G | NVIDIA NVFP4 | |
|---|---|---|---|
| Weights in GPU memory | 16.09 GiB | 19.55 GiB | **18 % smaller** |
| Decode, 1 user (tokens/s, community benchmark, median prompt) | 180.9 | 128.0 | **x1.41** |
| Decode per user at 4 / 8 users | 82.4 / 54.1 | 70.5 / 49.6 | x1.17 / x1.09 |
| Prompt processing, 2,048-token prompt (tokens/s) | ~3.4k | ~5.2k | x0.65 (slower) |
| GSM8K / HumanEval+ / MBPP+ / IFEval / MMLU-Pro | 90.83 / 88.41 / 76.72 / 82.62 / 77.7 | 91.10 / 89.02 / 76.19 / 84.29 / 77.3 | no significant difference on any |
| MGSM, multilingual maths | 81.9 | 84.2 | TR significantly lower |

Speed on English and code prompts; against NVFP4 with the same two engine options as our image, x1.31 at one user. MGSM ran both models on
our engine path (Model Runner V2, overlay, confidence stop). "No significant difference" is not a claim of equality.

## Models

| model | runtime |
|---|---|
| [Ternii/Qwen3.6-35B-A3B-TR2.25-G](https://huggingface.co/Ternii/Qwen3.6-35B-A3B-TR2.25-G) | vLLM, `ghcr.io/ternii-ai/tritq-qwen36:0.2` |
| [Ternii/Qwen3.6-35B-A3B-TR2.25-G-GGUF](https://huggingface.co/Ternii/Qwen3.6-35B-A3B-TR2.25-G-GGUF) | llama.cpp (Mac) |

## Licence

- vLLM plugin, Dockerfile and benchmark scripts: Apache License 2.0 (`LICENSE`; the plugin carries its own `LICENSE` and `NOTICE`).
- `runtime/vllm/tritq-qwen36/vllm028_overlay/`: modified vLLM files (a port of a vLLM pull request), Apache License 2.0, with their original copyright lines and change notices.
- `runtime/llama.cpp/`: MIT, like upstream llama.cpp.

The model weights are not in this repository; they are distributed on Hugging Face under the base models' licences.

This project is independent and is not affiliated with, sponsored or endorsed by Alibaba, NVIDIA, Apple or Hugging Face.

Publisher: Wai Chung · Contact: hello@ternii.com

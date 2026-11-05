# tritq-vllm: TR Quantisation runtimes

TR models are Qwen models whose weights are stored as packed ternary-digit (trit) codes with one 16-bit scale per group of 128
weights ("TR Quantisation"). Stock vLLM and llama.cpp cannot read this format; this repository holds the small runtime additions
that can:

- `runtime/vllm/`: the vLLM plugins for DGX Spark (Qwen3.6-35B-A3B on vLLM 0.28.0; Qwen3.8-Flash-Next on vLLM 0.32.0)
- `runtime/docker/`: the Dockerfiles for the images, one per model
- `runtime/llama.cpp/`: a patch series for llama.cpp on Apple silicon (Metal), with kernels for aarch64 CPUs
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

## Qwen3.8-Flash-Next TR2.57-G vs NVIDIA's NVFP4 checkpoint

Qwen3.8-Flash-Next has about 180B parameters in total. One DGX Spark per server, vLLM 0.32.0, text only, thinking off; NVIDIA's
`nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54` with weights unmodified, and a community NVFP4 recipe
(`salient-data/qwen38-next-dgx-recipe` @f309019, its published image and settings). **Each engine ran at its own serving settings.**
Full results, methods and caveats are on the [model card](https://huggingface.co/Ternii/Qwen3.8-Flash-Next-TR2.57-G).

| | TR2.57-G | NVIDIA NVFP4 | community NVFP4 recipe | |
|---|---|---|---|---|
| Weights in GPU memory, with the MTP draft head | 63.84 GiB | 75.63 GiB | 73.51 GiB | **16 % smaller** than NVIDIA's |
| Decode, 1 user (tokens/s, community benchmark, median prompt) | 84.3 | 40.5 | 71.9 | **x2.08** / **x1.17** |
| Decode, 6 users, total tokens/s | 95.2 | 62.0 | 85.2 | x1.54 / x1.12 |
| Prompt processing | not compared | | | |
| GSM8K / HumanEval+ / MBPP+ / IFEval | 92.11 / 92.07 / 79.10 / 83.73 | 91.85 / 93.90 / 78.84 / 84.84 | not run | no significant difference on any |
| MGSM / BFCL v4 | 89.96 / 74.04 | 89.13 / 72.20 | not run | no significant difference after correction |
| LiveCodeBench, thinking, 32k tokens (Qwen's FP8 original: 64.98) | 64.54 | 61.45 | not run | TR vs original: no significant difference; NVFP4 vs original: depends on the test family |

Speed on English prompts only (no NVFP4 comparison exists off English), with 2 comparator servers per comparator, so no p values for the
ratios; NVIDIA's checkpoint ran untuned, with its n-gram table read through our disk-backed lookup (not a stock setup). NVFP4 is
nominally ahead on HumanEval+, MBPP+ base and IFEval. Against Qwen's own FP8 release TR shows no significant difference on LiveCodeBench, GSM8K,
HumanEval+ and MBPP+. "No significant difference" is not a claim of equality: a 2-3 point
gap on HumanEval+ or LiveCodeBench would not have been detectable.

The Mac file for this model (`Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF`) is **experimental** and a different file (every translated weight
at 3 trits); the comparisons above do not apply to it.

## Models

| model | runtime |
|---|---|
| [Ternii/Qwen3.6-35B-A3B-TR2.25-G](https://huggingface.co/Ternii/Qwen3.6-35B-A3B-TR2.25-G) | vLLM, `ghcr.io/ternii-ai/tritq-qwen36:0.2` |
| [Ternii/Qwen3.6-35B-A3B-TR2.25-G-GGUF](https://huggingface.co/Ternii/Qwen3.6-35B-A3B-TR2.25-G-GGUF) | llama.cpp (Mac) |
| [Ternii/Qwen3.8-Flash-Next-TR2.57-G](https://huggingface.co/Ternii/Qwen3.8-Flash-Next-TR2.57-G) | vLLM, `ghcr.io/ternii-ai/tritq-qwen38:0.3` |
| [Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF](https://huggingface.co/Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF) | llama.cpp (Mac), experimental |

## Licence

- vLLM plugins, Dockerfiles and benchmark scripts: Apache License 2.0 (`LICENSE`; each plugin carries its own `LICENSE` and `NOTICE`).
- `runtime/vllm/tritq-qwen36/vllm028_overlay/`: modified vLLM files (a port of a vLLM pull request), Apache License 2.0, with their original copyright lines and change notices.
- `runtime/vllm/tritq-qwen38/patches/`: a vLLM pull request as a patch, unchanged from the community recipe that carries it, Apache License 2.0; the Dockerfile applies it to vLLM 0.32.0.
- `runtime/llama.cpp/`: MIT, like upstream llama.cpp.

The model weights are not in this repository; they are distributed on Hugging Face under the base models' licences.

This project is independent and is not affiliated with, sponsored or endorsed by Alibaba, NVIDIA, Apple or Hugging Face.

Publisher: Wai Chung · Contact: hello@ternii.com

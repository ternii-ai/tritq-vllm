# tritq-qwen36: TR Quantisation plugin for Qwen3.6-35B-A3B TR models (vLLM 0.28.0)

Adds the quantization method `"trit"` to vLLM 0.28.0: packed k-trit weights (2/3/4 trits, one fp16 scale per 128 weights) decoded by
Triton kernels, a decode-time MoE path (invalid expert ids masked, so it runs under Model Runner V2), and three MTP draft heads (sparse
all-language default, English-and-code, full). `vllm028_overlay/` adds the confidence-stop draft depth (port of vLLM PR #60068) to
vLLM 0.28.0's Model Runner V2; the Docker image applies it after checking the original files' sha256.

See `../../README.md` for the quick start and `src/tritq/__init__.py` for the options.

Tested: vLLM 0.28.0 (`vllm/vllm-openai:v0.28.0`), NVIDIA DGX Spark (GB10, sm_121), CUDA 13.0, one GPU. Other GPUs: not tested.
The default combination (Model Runner V2 + confidence stop + sparse draft head) is the measured release configuration (see the model card).

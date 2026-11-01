# Provenance of the files in tritq-vllm

Updated 2026-11-01 (first release: Qwen3.6-35B-A3B). "Ours" = written for this project by its publisher, Wai Chung. Upstream projects not listed per file are runtime dependencies only.

| package | file | origin | notes |
|---|---|---|---|
| tritq (Qwen3.6-35B-A3B, vLLM 0.28.0) | `fmt.py` | ours | packed trit layout (identical byte encoding to the llama.cpp GGUF types) |
| | `kern.py` | ours | Triton kernels: digit decode + `tl.dot`, MoE expert path |
| | `glue.py` | ours | fused decode MoE path (invalid expert ids masked to weight 0, needed by Model Runner V2), in-kernel split-K reduce, bf16 GEMV |
| | `sparse.py` | ours; algorithm of vllm#59740 (@stecasta, Apache-2.0) | sparse context-aware draft head for the packed-trit head; reimplementation, no lines copied; gather kernel follows the PR's layout |
| | `vllm_plugin.py` | ours | quantization method "trit", vLLM integration, draft-head options (sparse / English-and-code / full) |
| | `__init__.py` | ours | release defaults and public option names |
| vllm028_overlay (vLLM 0.28.0 files) | `vllm/v1/worker/gpu/spec_decode/confidence_stop.py` | vllm#60068 (@stecasta, Apache-2.0) via salient-data recipe patch 0010 @f309019 (PR head 029a811e) | class as in the patch + `cstop_config()` (env `TQ_CSTOP`) + two counters |
| | `vllm/v1/outputs.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu/cudagraph_utils.py`, `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py`, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` | vLLM 0.28.0 (Copyright contributors to the vLLM project, Apache-2.0), modified | the patch-0010 changes ported to 0.28 Model Runner V2, plus the trim in `execute_model` and a graph-size guard; each file carries a change notice; `SHA256SUMS.vllm-0.28.0-originals` pins the originals they replace |
| docker | `Dockerfile.qwen36-vllm028` | ours | |
| llama.cpp | `patches/0001`-`0004`, `apply.sh` | ours (MIT) | patch series on ggml-org/llama.cpp @ff5a233 |
| bench | `bench_oai.py` | ours | engine-neutral OpenAI-API decode benchmark |

Third-party benchmark harness: `bench/run_community_harness.sh` downloads `tools/bench_categories.py` from
https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark at commit 6ad1c8f (Apache-2.0, Copyright 2026 Tony DeAngelo
(Tech2Wild / 2Wild)) and checks its sha256. It is not redistributed here.

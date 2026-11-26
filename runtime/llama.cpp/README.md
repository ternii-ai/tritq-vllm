# llama.cpp with dense trit types (Mac / Metal, NVIDIA CUDA, Arm and x86 CPUs)

Patch series against **ggml-org/llama.cpp `ff5a23376725e5a889fa0bfb987b7541dc304da0`** (2026-10-18). Upstream LICENSE (MIT) unchanged;
our patches are MIT. The GGUF files of the TR models use tensor types that upstream llama.cpp does not have: they load only with these
patches.

| patch | what | default |
|---|---|---|
| `0001` ggml: TQ9_128 / TQ27_128 / TQ81_128 (CPU) | block types for 2 / 3 / 4-trit codes, one fp16 key per 128 weights (3.375 / 5.000 / 6.625 bits per weight), CPU reference, GGUF ids 44-46, gguf-py | always |
| `0002` ggml-metal: dense trit kernels | mat-vec, MUL_MAT_ID, GET_ROWS; native mat-mat tile | tile opt-in `GGML_METAL_TQD_MM=1` |
| `0003` ggml-metal: decode fusions for Qwen3.8-Flash-Next | GDN input prep folded into the GDN kernel; norm x gate in one dispatch | off; `GGML_METAL_OPT_GDNPREP=1 GGML_METAL_OPT_NGATED=1` |
| `0004` llama: SSD-backed MoE expert cache | experts stay in the GGUF file; a fixed cache of slots in GPU-visible memory; misses read from the SSD; output not bit-identical to the normal path: on Metal the cache splits the graph at each MoE layer, which changes which operations are fused, so values differ by float rounding (35B: greedy text the same on 5 test prompts, top-8 log-probabilities up to 0.79 apart over 32 tokens, as much as switching Metal's fusion off in the normal path; with `GGML_METAL_FUSION_DISABLE=1` on both, bit-identical); 16 GB-mode accuracy GSM8K 94.0 / HumanEval 92.1 / MBPP 87.6 vs normal path 95.0 / 92.7 / 87.2, no significant difference (exact McNemar); answer wording differs from the normal path on about a quarter of items, while correctness changes on a few | off; `LLAMA_MOE_SSD=1` |
| `0005` ggml-cpu: repacked kernels for the dense trit types | aarch64 CPU only: the TQ weights (including MoE experts) are repacked at load into a 4-row layout and run on NEON dotprod mat-vec / mat-mat kernels, with i8mm and SME2 mat-mat paths where the CPU has them; numerically equivalent to the 0001 CPU reference (float-level differences, not byte-identical); the repacked weights take more memory than the file (TQ9 +22 %, TQ27 +2.5 %, TQ81 +23 %) | on for aarch64; opt-out `GGML_TQ_NO_REPACK=1`, experts only `GGML_TQ_NO_MOE_REPACK=1`, per type `GGML_TQ_REPACK_MASK`, `GGML_TQ_NO_I8MM=1`, `GGML_TQ_NO_SME=1`, `GGML_TQ_SME_THREADS=n` (default 1) |
| `0006` ggml-cpu: repack 2-D weights with rows not a multiple of 4 | 0005 refused such tensors and left them on the 0001 scalar reference; they are now repacked with the last group of 4 padded with zero rows and only the real rows written back (MoE 3-D weights still need a multiple of 4). Matters for vocabularies like Qwen3's 151,669 tokens (e.g. Ternary Bonsai 1.7B re-packed with [`tools/`](tools/)), whose output layer runs on every generated token: there, decode ×2.7 on a Neoverse-N1-class CPU path; tensors with rows % 4 == 0 unchanged | on with 0005 |
| `0007` llama: 16 GB Mac defaults for the expert cache | only with `LLAMA_MOE_SSD=1`: (a) Qwen3.8-Flash-Next (qwen4exp) uses **8 routed experts per token** instead of the model's 10 (measured accuracy-neutral on the TR2.31-G file: GSM8K / HumanEval / MBPP / IFEval, no significant difference; not measured on the TR3-G file); (b) the experts of the next layer are predicted and read from the SSD while the GPU runs, with a GPU keep-busy filler so the GPU does not clock down while it waits; (c) a learned predictor `<model>.trpw` next to the first shard is loaded automatically and used for that read-ahead only (output unchanged; a missing or bad file falls back to the router with a warning); (d) the expert cache gives back the extra attention memory of windows above 4,096 tokens. MacBook Air M5 16 GB, Qwen3.8-Flash-Next TR2.31-G: about 12 tokens/s vs 3.6 for TR3-G on 0001-0006 | on with `LLAMA_MOE_SSD=1`; `LLAMA_MOE_EXPERTS=N` (1, 2, 4, 5, 6, 8, 9, 10; `0` = the model's own), `LLAMA_MOE_PRED=0` (no read-ahead), `LLAMA_MOE_PRED_K` (default 6), `LLAMA_MOE_PRED_W=<file>` (empty = no learned predictor), `GGML_METAL_MOE_SPIN=0` (no filler), `LLAMA_MOE_CACHE_REF_CTX` (default 4096, 0 = off); `--override-kv qwen4exp.expert_used_count=int:N` always wins |
| `0008` ggml-metal: multi-token mat-vec | mat-vec for 2-8 token rows (speculative-decoding verify passes, several users at once): each weight word is decoded once for all rows. Output is bit-identical to the stock mat-vec (test-backend-ops; answers identical to r3 on the 35B and Qwen3.8-Flash-Next); the mat-mat threshold is unchanged. M5 Max, llama-batched-bench, no drafting: total decode +8 to +24 % with 2-8 users, level with 1 user. With 1 user and n-gram drafting: level (mean -0.9 % over 5 alternating pairs on the 35B, within the run-to-run spread; at 1 user only ~0.4 % of the dense mat-vec calls carry 2-8 rows, so the kernel cannot cost more than a fraction of a percent there) | on; `GGML_METAL_TQD_EXT=0` turns it off |
| `0009` ggml-cuda: dense trit kernels | dequantisation (bit-identical to the CPU reference), mat-vec for 1-8 columns for MUL_MAT and MUL_MAT_ID (expert ids read on the device, so CUDA graphs keep working), GET_ROWS, and fp16 / bf16 / fp32 conversion for the cuBLAS path above 8 columns. Built and tested on a DGX Spark (GB10, sm_121, CUDA 13.0) only | always in CUDA builds |
| `0010` ggml-cpu: AVX2 dot product (x86) | the TQ x Q8_0 dot product with AVX2 instead of the 0001 scalar code; bitwise identical to it with the same compiler flags (checked by `tools/tq_exact/` in the patched tree). No repacked kernels on x86 | x86 with AVX2 |
| `0011` ggml-cuda: multi-token paths | `GGML_CUDA_TQ_MT`: grouped MoE mat-vec for 2-8 tokens without host syncs, exact against single-token decode on MoE layers (dense layers above 8 columns stay on cuBLAS unless `GGML_CUDA_TQ_MT_DENSE=1`). `GGML_CUDA_TQ_MM`: tensor-core mat-mat for prompts and long verify passes. **MM is value-changing**: outputs differ from `GGML_CUDA_TQ_MM=0`. DGX Spark, llama-bench pp2048, MM off -> on: 35B 1,387 -> 1,924, Qwen3.8-Flash-Next 607 -> 806 tok/s; decode level. Accuracy, MM on vs off (llama.cpp server, greedy, n-gram drafting): 35B GSM8K 90.83 vs 90.60, HumanEval+ 91.46 vs 92.68, MBPP+ 91.53 vs 91.27; Qwen3.8-Flash-Next GSM8K 92.80 vs 93.25; no significant difference (all p ≥ 0.33) | both on; `GGML_CUDA_TQ_MT=0`, `GGML_CUDA_TQ_MM=0` |
| `0012` ggml-cuda: mat-mat v4 | `GGML_CUDA_TQ_MM_V2`: a second tensor-core mat-mat kernel for the dense (non-MoE) layers, used when `GGML_CUDA_TQ_MM` is on. **Value-changing**: outputs differ from `GGML_CUDA_TQ_MM_V2=0` (0011's mat-mat). DGX Spark, llama-bench: prompt processing about 3 % faster than 0011's mat-mat on both models (pp512 / pp2048 ×1.03 on the 35B, ×1.025 on Qwen3.8-Flash-Next), decode level. Accuracy, 0012 vs 0011 mat-mat (llama.cpp server, greedy, n-gram drafting): 35B GSM8K 90.37 vs 90.83, HumanEval+ 92.07 vs 91.46, MBPP+ 91.27 vs 91.53; Qwen3.8-Flash-Next GSM8K 92.80 vs 92.80; no significant difference (all p ≥ 0.24) | on; `GGML_CUDA_TQ_MM_V2=0` gives 0011's mat-mat |
| `0013` ggml-cuda: streamed mat-vec v4 | `GGML_CUDA_TQ_MMV`: a persistent decode mat-vec that loads the next rows while it decodes the current ones; same arithmetic in the same order as 0009's mat-vec, so **outputs are bit-identical** to `GGML_CUDA_TQ_MMV=1` (test-backend-ops, direct output comparison on 13 shapes, 35B perplexity per chunk, greedy text on both models). Used for dense mat-vec with 1 column (all three types) and 2 columns (TQ27 / TQ81); experts and 3-8 columns stay on 0009's kernel. DGX Spark, llama-bench tg128, r5 -> r6: 35B TR2.25-G 85.5 -> 91.7 tok/s (×1.07), 35B TR2.27-GK ×1.10, Qwen3.8-Flash-Next ×1.09 | on; `GGML_CUDA_TQ_MMV=1` |
| `0014` ggml-cuda: mat-mat v5 | `GGML_CUDA_TQ_MM_V2=3`: a pipelined int8 tensor-core mat-mat for the dense (non-MoE) layers; bit-identical to the int8 variant `GGML_CUDA_TQ_MM_V2=2` | opt-in |
| `0015` ggml-cuda: one activation scale per 128 | `GGML_CUDA_TQ_MM_V2=4`: 0014's kernel with the activations quantised to int8 with one scale per 128 values (one trit block) instead of one per 32, so each block needs one float step instead of four | made the default by 0016 |
| `0016` ggml-cuda: per-128 mat-mat by default | `GGML_CUDA_TQ_MM_V2` default 1 -> 4. **Value-changing**: outputs differ from `GGML_CUDA_TQ_MM_V2=1` (r5) for prompts and verify passes of 8 or more tokens; single-token decode and the experts are unaffected. DGX Spark, llama-bench, r5 -> r6 (0013-0016): prompt processing ×1.06 on the 35B (pp512 2,016 -> 2,132, pp2048 1,997 -> 2,120 tok/s), ×1.05 on Qwen3.8-Flash-Next (pp2048 826 -> 864). Accuracy, per-128 vs r5 (llama.cpp server, greedy, n-gram drafting): 35B GSM8K 90.83 vs 90.37, HumanEval+ 92.07 vs 92.07, MBPP+ 91.01 vs 91.27; Qwen3.8-Flash-Next GSM8K 92.87 vs 92.80 (n = 1,319); no significant difference (all p ≥ 0.42) | on; `GGML_CUDA_TQ_MM_V2=1` gives r5's mat-mat |
| `0017` llama: expert cache clamps the micro-batch | with `LLAMA_MOE_SSD=1`, a `-ub` larger than the cache takes (e.g. Ollama's 1024 or 2048) used to abort the server at start; it is now lowered to the limit with a warning, and the output is byte-identical to the same request at `-ub` = the limit (greedy text and top-8 log-probabilities, 35B, M5 Max). `LLAMA_MOE_CACHE_MAX_BATCH` now defaults to 512 in SSD mode (was 32), so `LLAMA_MOE_SSD=1` alone works at llama-server's default `-ub 512`; an explicit value is used as before. No effect without `LLAMA_MOE_SSD=1` | always |
| `0018` ggml-metal: expert cache, one command buffer per graph in every context | Metal only, async expert cache (`LLAMA_MOE_SSD=1 LLAMA_MOE_ASYNC=1`, the 16 GB mode): only the first context was switched to one command buffer per graph, so a second context in the same process could read the selected experts too early; `llama-bench`, which makes a new context per test, aborted on its tg test after a pp test. Every context is now switched. llama-server and llama-cli use one context: no output change (llama-server, 35B, 16 GB settings: tokens and top-8 log-probabilities bit-identical to 0017) | always |
| `0019` ggml-metal: DeltaNet decode fusions on by default | the two fusions of 0003 (gated-delta-net input preparation; RMS norm × weight × gate) are now on by default on Metal. **Outputs are bit-identical** to r6 (top-8 log-probabilities over 40 greedy steps at 12 / 4,180 / 14,900-token prompts, 35B TR2.25-G and TR2.27-GK, also in the 16 GB expert-cache mode) | on; `GGML_METAL_OPT_GDNPREP=0`, `GGML_METAL_OPT_NGATED=0` |
| `0020` ggml-metal: gated-delta-net state read from the cache | in a one-token decode graph, the gated-delta-net kernel reads each layer's state straight from the cache instead of copying it out first (2 MiB per layer per token). Bit-identical (as 0019) | on; `GGML_METAL_OPT_RSREAD=0` |
| `0021` ggml-metal: conv state in one kernel | the conv-state gather, concat and write-back of each DeltaNet layer run as one kernel. Bit-identical (as 0019) | on; `GGML_METAL_OPT_CONVRS=0` |
| `0022` ggml-metal: pre-encode the next graph | right after a decode graph is committed, the same graph is encoded again (in the same encoder ranges, so the same kernels) for the next token; it is used only if the graph is unchanged. Bit-identical (as 0019; test-backend-ops 474/474 on the trit types and 938/938 on the touched ops). Off with the async expert cache (16 GB mode), the profiler and captures. M5 Max, 35B TR2.25-G, llama-server with n-gram drafting, r6 -> r7 (0019-0022): decode ×1.31 (×1.28-1.33, 4 passes on a shared machine); prompts unchanged. 16 GB MacBook Air (expert-cache mode, only 0019-0021 act): ×1.05 | on; `GGML_METAL_PREENCODE=0` |

## Ready-built binaries

No build needed: the [`llama.cpp-tr-b11721-r7` release](https://github.com/ternii-ai/tritq-vllm/releases/tag/llama.cpp-tr-b11721-r7) has this series
(0001-0022 on `ff5a233`) built for **macOS arm64** (Apple silicon, macOS 13.3+, Metal; CPU code for M1 / M2-M3 / M4+ picked at run time) and
**Linux arm64** (glibc 2.39+, e.g. Ubuntu 24.04 / DGX OS; CPU code for armv8.0 ... armv9.2 + SME picked at run time, plus CUDA for the
DGX Spark, sm_121, which needs the CUDA 13 runtime; without it the CPU is used). There is no x86-64 archive: build from source with `apply.sh`.
Earlier releases: [r6](https://github.com/ternii-ai/tritq-vllm/releases/tag/llama.cpp-tr-b11721-r6) (0001-0018, macOS and Linux arm64),
[r5](https://github.com/ternii-ai/tritq-vllm/releases/tag/llama.cpp-tr-b11721-r5) (0001-0012, macOS and Linux arm64),
[r4](https://github.com/ternii-ai/tritq-vllm/releases/tag/llama.cpp-tr-b11721-r4) (0001-0011, macOS and Linux arm64; the Qwen3.8-Flash-Next 16 GB installer uses it),
[r3](https://github.com/ternii-ai/tritq-vllm/releases/tag/llama.cpp-tr-b11721-r3) (0001-0007, macOS arm64),
[r2](https://github.com/ternii-ai/tritq-vllm/releases/tag/llama.cpp-tr-b11721-r2) (0001-0006, macOS and Linux arm64, CPU only).
Each archive has `llama-server` (with the web UI), `llama-cli`, `llama-completion`, `llama-bench`, `llama-perplexity` and `test-backend-ops`.
The run lines below work with them as they are (`./llama-server` in the unpacked folder instead of `./build/bin/llama-server`).
On macOS, an archive downloaded with a web browser needs `xattr -dr com.apple.quarantine <folder>` once. The archives are made by
[`release/`](release/) (`build_macos_arm64.sh`, `build_linux_arm64.sh`); `--version` shows a local commit made by `apply.sh`, not an upstream one.

## Trained ternary models (`tools/`)

[`tools/ternary_gguf_to_tq9.py`](tools/ternary_gguf_to_tq9.py) re-packs a model that was trained ternary (weights already -1 / 0 / +1 times a
scale, e.g. PrismML's Ternary Bonsai) into TQ9_128, so it runs on the 0005 / 0006 aarch64 CPU kernels. It needs the model's F16 GGUF and a
ternary GGUF of it (Q2_0, TQ1_0 or TQ2_0 tensors mark which weights to re-pack; everything else is copied), and the `gguf-py` of a tree made by
`apply.sh` (Python 3 with numpy and PyYAML; Ubuntu 22.04's `python3-numpy python3-yaml` work). TQ9_128 takes 3.375 bits per weight, so the file is larger than a 2-bit ternary GGUF.

Ternary Bonsai 1.7B, one machine (DGX Spark, 4 Cortex-A725 cores, each build limited to the dotprod + fp16 CPU code a Neoverse-N1 runs;
`llama-bench` tg128 at 4 threads / pp512, two passes; KL divergence vs the F16 file on WikiText-2, 20 × 512 tokens):

| file | build | size | tok/s generate / prompt | KLD |
|---|---|---|---|---|
| `Q2_0_g64` (PrismML) | upstream llama.cpp 9f0e5098a | 0.49 GB | 21.7 / 27.8 | 0.0002 |
| `PQ2_0` (PrismML) | PrismML's llama.cpp fork 6447906bf | 0.46 GB | 27.3 / 35.6 | 0.0010 |
| TQ9_128 from this tool | this series, 0001-0006 | 0.73 GB | 41.8 / 57.2 | 0.0002 |

One CPU type and one model; not measured on a real Neoverse-N1 (the A725 cores run upstream's files 18-46 % faster than published N1 figures,
with the same ranking).

## Build

```bash
./apply.sh llama.cpp-tr && cd llama.cpp-tr
# macOS (Metal is on by default on Apple silicon)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server llama-cli llama-bench llama-perplexity test-backend-ops
# Metal gate (needs ~14 GB free, ~2 min): must print "402/402 tests passed" and list tq9_128 / tq27_128 / tq81_128 cases.
# The Metal backend is named MTL0; a run that prints only "Skipping" tested nothing.
./build/bin/test-backend-ops -b MTL0 -o MUL_MAT,MUL_MAT_ID,GET_ROWS -p 'tq(9|27|81)_128'
```

## Run (Mac)

```bash
# 35B, fully in memory (Macs with >= 24 GB; 32 GB comfortable)
./build/bin/llama-server -m Qwen3.6-35B-A3B-TR2.25-G.gguf -c 4096 -fa on --jinja
# 35B on a 16 GB Mac: expert cache of 8,400 MiB (experimental; ~35 tok/s on a MacBook Air M5 16 GB with r3; r4 level with r3 there, same answers). One-line setup instead:
#   curl -fsSL https://raw.githubusercontent.com/ternii-ai/tritq-vllm/main/mac/qwen36-35b.sh | bash
caffeinate -ims env LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB=8400 LLAMA_MOE_CACHE_MAX_BATCH=512 LLAMA_MOE_ASYNC=1 GGML_SCHED_COPY_CB_FIRST=1 ./build/bin/llama-server -m Qwen3.6-35B-A3B-TR2.25-G.gguf -c 4096 -fa on --jinja -np 1
# Qwen3.8-Flash-Next TR2.31-G on a 16 GB Mac (8 shards + the .trpw predictor in one folder; needs 0007). One-line setup instead:
#   curl -fsSL https://raw.githubusercontent.com/ternii-ai/tritq-vllm/main/mac/qwen38-16gb.sh | bash
caffeinate -ims env LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB=6750 LLAMA_MOE_CACHE_MAX_BATCH=512 LLAMA_MOE_ASYNC=1 GGML_SCHED_COPY_CB_FIRST=1 \
GGML_METAL_OPT_GDNPREP=1 GGML_METAL_OPT_NGATED=1 GGML_TQ_NO_REPACK=1 \
./build/bin/llama-server -m Qwen3.8-Flash-Next-TR2.31-G-00001-of-00008.gguf -c 4096 -fa on --jinja -np 1 --cache-ram 0 --reasoning off
# Qwen3.8-Flash-Next TR3-G (experimental; 10 shards, point -m at the first). Expert cache 35000 (64 GB and up), 13501 (32-48 GB),
# 6750 (24 GB), 5900 (16 GB); measured by capping memory on a 128 GB M5 Max, not on real smaller Macs. 0003's two fusions are on in every measurement.
LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB=35000 LLAMA_MOE_CACHE_MAX_BATCH=512 LLAMA_MOE_ASYNC=1 GGML_SCHED_COPY_CB_FIRST=1 \
GGML_METAL_OPT_GDNPREP=1 GGML_METAL_OPT_NGATED=1 GGML_TQ_NO_REPACK=1 \
./build/bin/llama-server -m Qwen3.8-Flash-Next-TR3-G-00001-of-00010.gguf -c 4096 -fa on --jinja -np 1 --cache-ram 0
```

The 35B expert-cache measurements and the Qwen3.8 replay and accuracy runs used patches 0001-0004 (the Qwen3.8 server runs used
the full series with `GGML_TQ_NO_REPACK=1`); 0005 (on by default on aarch64) has not been timed with the expert
cache, so those two command lines turn it off (`GGML_TQ_NO_REPACK=1`), which gives the measured configuration. Speeds,
memory and accuracy for it are on the `Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF` model card.

`--cache-ram 0` turns off llama-server's host-memory prompt cache (default up to 8 GiB). With it on, the Qwen3.8 server's memory grew by
about 235 MiB per request until that cache filled (about 35 requests): up to 8 GiB of ordinary memory on top of the expert cache and
the model's wired memory. With it off, the server's footprint stayed flat. The 35B memory and accuracy figures came from
llama-completion and a replay tool; llama-server was only timed (4 requests), so its prompt cache was never part of the 35B memory
figures. On a tight memory budget, adding `--cache-ram 0` to the 35B lines is the safe choice.

The cache settings above are the measured configurations; they were measured at 4k context only, and single-sequence only (hence `-np 1`; llama-server otherwise defaults to 4 slots). With `LLAMA_MOE_SSD=1` the
micro-batch is capped at `LLAMA_MOE_CACHE_MAX_BATCH` (512 by default since 0017): a larger `-ub` is lowered to it with a warning. Other cache knobs (`LLAMA_MOE_SSD_THREADS`, `LLAMA_MOE_SSD_CHUNK`, `LLAMA_MOE_PRELOAD`,
`LLAMA_MOE_ASYNC_*`, `LLAMA_MOE_STATS=<file>`) are for tuning and diagnostics. `LLAMA_MOE_ASYNC=1` needs `GGML_SCHED_COPY_CB_FIRST=1`
(0004 aborts on ASYNC without it). On the 35B 16 GB line async is measured output-identical to the sync cache path (greedy 8/8, raw
logits) and +30 % sustained decode; it was measured at `-np 1` with sequential requests, 4k context and short prompts only — concurrency,
cancelled requests and long prompts are untested.

Qwen3.6 runs in thinking mode by default with `--jinja`: a short question can spend 1,000+ tokens in `reasoning_content` before
any answer (3 of 4 test requests hit a 1,024-token limit). For a quick test, start the server with `--reasoning off` (at ff5a233 this
replaces the deprecated `--chat-template-kwargs '{"enable_thinking":false}'`), or send `"chat_template_kwargs": {"enable_thinking": false}`
in the request, or raise `max_tokens`. These switches are standard llama.cpp options; we did not run them in our measurements.

## Known limitations

- **0007 changes the expert-cache path for every model that uses it** (`LLAMA_MOE_SSD=1`): the GPU filler runs for the 35B
  too (output unchanged in a 4-prompt check; the read-ahead itself applies to Qwen3.8-Flash-Next only), and the 8-expert default applies to the Qwen3.8-Flash-Next TR3-G file as well, where its
  accuracy has not been measured: set `LLAMA_MOE_EXPERTS=0` to keep 10 experts there. `LLAMA_MOE_PRED=0 GGML_METAL_MOE_SPIN=0
  LLAMA_MOE_EXPERTS=0` gives 0004's behaviour back. Run Macs on 0007 under `caffeinate -ims` (or another power assertion): without it the
  base M5's GPU clocks down and decode is much slower.
- **CUDA (0009, 0011-0016) has been built and tested on a DGX Spark (GB10, sm_121, CUDA 13.0) only.** There, with n-gram drafting on code
  edits, the 35B is level with Q4_K_M since 0012 (149.9 vs 148.0 tok/s, 6 passes; 139.2 with 0011's mat-mat); on maths, chat and documents it is ahead. `GGML_CUDA_TQ_MM=0
  GGML_CUDA_TQ_MT=0` gives 0009's plain kernels back.
- The fast CPU path (0005) is aarch64 only. x86 gets 0010's AVX2 dot product (exact, not repacked); its speed is not published here.
- `test-backend-ops` does not exercise 0005's repacked path (it never places weights in the CPU repack buffer), so a passing CPU run there does not test it.
- **`LLAMA_MOE_CACHE_MAX_BATCH`** defaulted to 32 in 0004-0016, below llama.cpp's default 512-token micro-batch, and with
  `LLAMA_MOE_SSD=1` any `-ub` above it aborted at context creation. Since 0017 the SSD mode defaults to 512 and lowers a larger `-ub`
  to the limit instead (larger prefill batches still need the knob raised, at the cost of GPU memory). An unparsable value (read as 0)
  still aborts.
- Up to r5 the 0004 patch's commit message said "Output is unchanged versus the fully resident model"; that was not exact: on Metal the
  cache path differs from the fully resident path by float rounding from operation fusion (see the 0004 row above). r6 corrects the
  message; the code is unchanged.
- GGUF type ids 44-46 are used only by this series and may collide with future upstream llama.cpp types.

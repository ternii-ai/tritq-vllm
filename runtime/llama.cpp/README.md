# llama.cpp with dense trit types (Mac / Metal)

Patch series against **ggml-org/llama.cpp `ff5a23376725e5a889fa0bfb987b7541dc304da0`** (2026-10-18). Upstream LICENSE (MIT) unchanged;
our patches are MIT. The GGUF files of the TR models use tensor types that upstream llama.cpp does not have: they load only with these
patches.

| patch | what | default |
|---|---|---|
| `0001` ggml: TQ9_128 / TQ27_128 / TQ81_128 (CPU) | block types for 2 / 3 / 4-trit codes, one fp16 key per 128 weights (3.375 / 5.000 / 6.625 bits per weight), CPU reference, GGUF ids 44-46, gguf-py | always |
| `0002` ggml-metal: dense trit kernels | mat-vec, MUL_MAT_ID, GET_ROWS; native mat-mat tile | tile opt-in `GGML_METAL_TQD_MM=1` |
| `0003` ggml-metal: decode fusions for Qwen3.8-Flash-Next (a model not in this release) | GDN input prep folded into the GDN kernel; norm x gate in one dispatch | off; `GGML_METAL_OPT_GDNPREP=1 GGML_METAL_OPT_NGATED=1` |
| `0004` llama: SSD-backed MoE expert cache | experts stay in the GGUF file; a fixed cache of slots in GPU-visible memory; misses read from the SSD; output numerically equivalent to the normal path (float-level differences, not byte-identical; measured on the 35B, public series); 16 GB-mode accuracy GSM8K 94.0 / HumanEval 92.1 / MBPP 87.6 vs normal path 95.0 / 92.7 / 87.2, no significant difference (exact McNemar); answer wording differs from the normal path on about a quarter of items, while correctness changes on a few | off; `LLAMA_MOE_SSD=1` |

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
# 35B in a 16 GB budget: expert cache of 8,400 MiB (experimental; measured by capping memory on a 128 GB M5 Max, not on a real 16 GB Mac)
LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB=8400 LLAMA_MOE_CACHE_MAX_BATCH=512 LLAMA_MOE_ASYNC=1 GGML_SCHED_COPY_CB_FIRST=1 ./build/bin/llama-server -m Qwen3.6-35B-A3B-TR2.25-G.gguf -c 4096 -fa on --jinja -np 1
```

The cache settings above are the measured configurations; they were measured at 4k context only, and single-sequence only (hence `-np 1`; llama-server otherwise defaults to 4 slots). The cache path asserts at most 512
tokens per micro-batch (`-ub` <= 512). Other cache knobs (`LLAMA_MOE_SSD_THREADS`, `LLAMA_MOE_SSD_CHUNK`, `LLAMA_MOE_PRELOAD`,
`LLAMA_MOE_ASYNC_*`, `LLAMA_MOE_STATS=<file>`) are for tuning and diagnostics. `LLAMA_MOE_ASYNC=1` needs `GGML_SCHED_COPY_CB_FIRST=1`
(0004 aborts on ASYNC without it). On the 35B 16 GB line async is measured output-identical to the sync cache path (greedy 8/8, raw
logits) and +30 % sustained decode; it was measured at `-np 1` with sequential requests, 4k context and short prompts only — concurrency,
cancelled requests and long prompts are untested.

Qwen3.6 runs in thinking mode by default with `--jinja`: a short question can spend 1,000+ tokens in `reasoning_content` before
any answer (3 of 4 test requests hit a 1,024-token limit). For a quick test, start the server with `--reasoning off` (at ff5a233 this
replaces the deprecated `--chat-template-kwargs '{"enable_thinking":false}'`), or send `"chat_template_kwargs": {"enable_thinking": false}`
in the request, or raise `max_tokens`. These switches are standard llama.cpp options; we did not run them in our measurements.

## Known limitations

- **CUDA is not supported** by this patch series: the TQ types run on the CPU and on Metal only.
- **`LLAMA_MOE_CACHE_MAX_BATCH` defaults to 32 in 0004**, below llama.cpp's default 512-token micro-batch: `LLAMA_MOE_SSD=1` without
  the knob aborts at context creation (the run lines above set it). A value below `-ub`, or an unparsable one, aborts the same way.
- The 0004 patch's own commit message says "Output is unchanged versus the fully resident model"; for the 35B this is not exact
  (numerically equivalent, not byte-identical; see the table above).
- GGUF type ids 44-46 are used only by this series and may collide with future upstream llama.cpp types.

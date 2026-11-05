# Third-party notices

Updated 2026-11-05 for Qwen3.6-35B-A3B and Qwen3.8-Flash-Next.

## 1. vLLM pull request #60068: confidence stop for autoregressive drafting (IN this release)

- **Upstream:** https://github.com/vllm-project/vllm/pull/60068, author @stecasta, Apache-2.0 (open PR; ported at head 029a811e).
- **As carried by:** salient-data/qwen38-next-dgx-recipe @f309019, `recipe/patches/0010-vllm60068-confidence-stop-for-autoregressive-drafting.patch`,
  Copyright 2026 Salient Data, Apache-2.0.
- **In this distribution:**
  - Qwen3.8-Flash-Next (vLLM 0.32.0): `tritq-qwen38/patches/0010-…patch`, byte-identical to the recipe's file (md5
    e9631fa9a9f2fccf78be0306d276331d), provenance header kept. The Dockerfile applies it with `patch --fuzz=0` and keeps the patch,
    its sha256 and the list of modified vLLM files in the image (`/opt/tritq-qwen38/patches/`).
  - Qwen3.6-35B-A3B (vLLM 0.28.0): `tritq-qwen36/vllm028_overlay/vllm/v1/…`, a port of the same patch onto vLLM 0.28.0's Model Runner
    V2: `confidence_stop.py` (the patch's class, plus a `cstop_config()` reader for the environment variable `TQ_CSTOP` and two
    counters) and six modified vLLM 0.28.0 files (`outputs.py`, `core/sched/scheduler.py`, `worker/gpu/cudagraph_utils.py`,
    `worker/gpu/model_runner.py`, `worker/gpu/spec_decode/speculator.py`, `worker/gpu/spec_decode/autoregressive/speculator.py`).
    Port changes beyond the PR: the verify trim is done in `execute_model` (0.28 picks the CUDA graph before `prepare_inputs`), and a
    graph-size guard for the batch-size draft schedule.
- **Terms complied with:** every modified file keeps "SPDX-License-Identifier: Apache-2.0" and "SPDX-FileCopyrightText: Copyright
  contributors to the vLLM project" (Apache-2.0 section 4(c)); each carries "Modified by Wai Chung, 2026-10 …" (section 4(b)); the NOTICE
  file carries the vLLM and Salient Data NOTICE lines (section 4(d)).

## 2. salient-data/qwen38-next-dgx-recipe patch 0011: skinny-GEMM rows (IN this release)

- `recipe/patches/0011-skinny-gemm-rows.patch` @f309019, "written for this recipe" (no vLLM PR), Copyright 2026 Salient Data,
  Apache-2.0.
- `tritq125/vllm_plugin.py` `_patch_skinny_fill()` follows its fill loop line for line (the same rule: every missing row count gets the
  next larger plan's configuration), applied at run time over every capability table rather than as an edit to vLLM's SM121 table. It
  is a derivative of that patch and is credited in NOTICE and in the function's docstring.

## 3. vLLM pull request #59740: context-aware sparse LM head (algorithm reimplemented)

- https://github.com/vllm-project/vllm/pull/59740, author @stecasta, Apache-2.0 (also carried by the recipe as patch 0013).
- `tritq/sparse.py` and `tritq125/sparse.py` reimplement the algorithm for our packed-trit head: a fixed vocabulary prefix with exact logits, plus a per-token set
  of rows chosen by a low-rank projection of the head, with exact logits for those rows. The gather kernel follows the PR's
  row-gather GEMV layout. No lines were copied. Credited as a reimplementation; Apache-2.0 places no obligation on reimplementing an
  algorithm, but the credit is kept for verifiability.

## 4. Atlas inference engine: idea only

- The reduced-vocabulary draft head (`TRITQ_DRAFT_HEAD=english`, `tritq125/lever1.py`, and the draft-vocabulary path of both
  `vllm_plugin.py` files) implements an idea that Atlas exposes as `--mtp-vocab` and that FR-Spec and VocabTrim publish. **No Atlas
  code is in this distribution and none was adapted.** Atlas is AGPL-3.0; MiaAI `sm121_varlen.py` (AGPL-3.0-or-later) is not used
  either.

## 5. Community benchmark harness (not redistributed)

- `bench/run_community_harness.sh` downloads `tools/bench_categories.py` from https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark
  at commit 6ad1c8f (Apache-2.0, Copyright 2026 Tony DeAngelo (Tech2Wild / 2Wild)) and checks its sha256. It is not included here.

## 6. Datasets used to calibrate the published weights

Not part of this software distribution; credited in the model repositories' NOTICE files: OpenCodeInstruct (`nvidia/OpenCodeInstruct`,
CC BY 4.0, NVIDIA); MATH (`EleutherAI/hendrycks_math` training split, MIT, Copyright (c) 2021 Dan Hendrycks); GSM8K (`openai/gsm8k`
training split, MIT, Copyright (c) 2021 OpenAI); OpenAssistant oasst2 (`OpenAssistant/oasst2`, Apache-2.0; Qwen3.8-Flash-Next only).

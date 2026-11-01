#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Runs the community single-Spark speed harness (tonyd2wild bench_categories.py) unchanged against an OpenAI-compatible server.
# The harness is downloaded at a pinned commit and checked by sha256; it is NOT redistributed here
# (Apache-2.0, Copyright 2026 Tony DeAngelo (Tech2Wild / 2Wild), https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark).
# usage: run_community_harness.sh <base url, e.g. http://127.0.0.1:8000> <served model name> <label> [runs, default 3]
# Protocol used for the published numbers: one discarded warm-up run, then 3 runs; figure = median prompt, median of the 3 runs.
# Note: at this commit the five "summary" prompts read bench-docs/*.md, which the repo does not contain, so they run without a
# document (as they do for everyone using this commit).
set -euo pipefail
BASE=$1; MODEL=$2; LABEL=$3; RUNS=${4:-3}
COMMIT=6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0
SHA=5d9b6954d0c87be8226aca86297704ca8c968a3863353d5c267480bddf06e0d3
D=${HARNESS_DIR:-$HOME/.cache/tritq-bench}; mkdir -p "$D"; F="$D/bench_categories_$COMMIT.py"
if [ ! -f "$F" ]; then
  curl -fsSL "https://raw.githubusercontent.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/$COMMIT/single-spark-vllm-tp1/tools/bench_categories.py" -o "$F.tmp"
  mv "$F.tmp" "$F"
fi
echo "$SHA  $F" | shasum -a 256 -c -
mkdir -p results
python3 "$F" "$BASE" "$MODEL" "$LABEL" --tag warmup > /dev/null
for i in $(seq 1 "$RUNS"); do python3 "$F" "$BASE" "$MODEL" "$LABEL" --tag "run$i"; done
# Concurrency 6 (the harness's multi-user figure): add --concurrency 6 and report total tokens/s.

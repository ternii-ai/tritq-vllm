#!/bin/bash
# SPDX-License-Identifier: MIT
# Clones ggml-org/llama.cpp at the pinned commit and applies patches/000*.patch with git am.
# usage: ./apply.sh [target dir, default ./llama.cpp-tr]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); BASE=$(cat "$HERE/PINNED_BASE"); DST=${1:-llama.cpp-tr}
git clone https://github.com/ggml-org/llama.cpp "$DST"
cd "$DST"; git checkout -q "$BASE"; git checkout -q -b tr-trit
git -c user.name="tr-build" -c user.email="tr-build@invalid" am "$HERE"/patches/000*.patch
git log --oneline -5

#!/bin/bash
# SPDX-License-Identifier: MIT
# Builds the Linux arm64 (CPU) release archive inside ubuntu:24.04 (glibc 2.39; gcc 14, needed for the armv9.2 + SME variants):
# CPU code as armv8.0 ... armv9.2 variants picked at run time (GGML_BACKEND_DL), OpenMP off (no libgomp dependency),
# BoringSSL linked statically, rpath $ORIGIN.
# usage (on an arm64 host with Docker): ./build_linux_arm64.sh [work dir, default ./build-linux]
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); W=$(mkdir -p "${1:-build-linux}" && cd "${1:-build-linux}" && pwd)
docker run --rm -v "$HERE:/series:ro" -v "$W:/w" -e HF_UI_VERSION=b11721 ubuntu:24.04 bash -euxo pipefail -c '
  apt-get update -qq && apt-get install -y -qq git cmake build-essential gcc-14 g++-14 ca-certificates > /dev/null
  cd /w && rm -rf llama.cpp-tr && bash /series/apply.sh llama.cpp-tr && cd llama.cpp-tr
  CC=gcc-14 CXX=g++-14 cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_RPATH="\$ORIGIN" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_BORINGSSL=ON \
    -DCMAKE_C_FLAGS="-ffile-prefix-map=/w=/src" -DCMAKE_CXX_FLAGS="-ffile-prefix-map=/w=/src"
  cmake --build build -j --target llama-server llama-cli llama-bench llama-perplexity llama-completion test-backend-ops
  bash /series/release/pack.sh llama.cpp-tr-linux-arm64 /w/llama.cpp-tr /w
  chown -R '"$(id -u):$(id -g)"' /w'

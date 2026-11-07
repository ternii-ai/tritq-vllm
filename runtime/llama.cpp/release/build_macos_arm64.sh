#!/bin/bash
# SPDX-License-Identifier: MIT
# Builds the macOS arm64 release archive: Metal library embedded, CPU code as apple_m1 / apple_m2_m3 / apple_m4 variants picked at
# run time (GGML_BACKEND_DL), BoringSSL linked statically, rpath @loader_path, deployment target 13.3.
# usage: ./build_macos_arm64.sh [work dir, default ./build-macos]
set -euxo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=$(mkdir -p "${1:-build-macos}" && cd "${1:-build-macos}" && pwd)
export HF_UI_VERSION=b11721   # web UI of upstream's release for the pinned base commit
cd "$W"; rm -rf llama.cpp-tr; bash "$HERE/../apply.sh" llama.cpp-tr; cd llama.cpp-tr
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_RPATH='@loader_path' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_BORINGSSL=ON \
  -DCMAKE_C_FLAGS="-ffile-prefix-map=$W=/src" -DCMAKE_CXX_FLAGS="-ffile-prefix-map=$W=/src" -DCMAKE_OBJC_FLAGS="-ffile-prefix-map=$W=/src"
cmake --build build -j --target llama-server llama-cli llama-bench llama-perplexity llama-completion test-backend-ops
bash "$HERE/pack.sh" llama.cpp-tr-macos-arm64 "$W/llama.cpp-tr" "$W"

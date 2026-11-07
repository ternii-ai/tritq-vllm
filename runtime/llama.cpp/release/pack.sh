#!/bin/bash
# SPDX-License-Identifier: MIT
# pack.sh <archive name> <llama.cpp-tr tree> <output dir>: binaries, licences, NOTICE and README.txt into <name>.tar.gz.
set -euo pipefail
N=$1; T=$2; O=$3; HERE=$(cd "$(dirname "$0")" && pwd); B=$T/build/bin; Q=$O/$N
rm -rf "$Q" && mkdir -p "$Q"
cp -a $B/llama-server $B/llama-cli $B/llama-completion $B/llama-bench $B/llama-perplexity $B/test-backend-ops "$Q/"
cp -a $B/*.so* "$Q/" 2>/dev/null || true; cp -a $B/*.dylib "$Q/" 2>/dev/null || true
cp $T/LICENSE "$Q/LICENSE"; cp "$HERE/../LICENSE" "$Q/LICENSE-TR"; cp $T/build/_deps/boringssl-src/LICENSE "$Q/LICENSE-boringssl"
cp $T/vendor/cpp-httplib/LICENSE "$Q/LICENSE-httplib"; cp $T/licenses/LICENSE-jsonhpp "$Q/"
cp "$HERE/NOTICE" "$HERE/README.txt" "$Q/"
(cd "$O" && COPYFILE_DISABLE=1 tar czf "$N.tar.gz" "$N" && ls -la "$N.tar.gz")

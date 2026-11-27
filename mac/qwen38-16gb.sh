#!/bin/bash
# SPDX-License-Identifier: MIT
# One-line setup for Qwen3.8-Flash-Next (TR Quantisation) on an Apple silicon Mac with 16 GB or more:
#   curl -fsSL https://raw.githubusercontent.com/ternii-ai/tritq-vllm/main/mac/qwen38-16gb.sh | bash
# Downloads the ready-built llama.cpp-tr and the model (about 90 GB) into ~/qwen38-tr, checks every file's SHA-256,
# then starts a local chat server and opens it in your browser. Run the same line again to restart; finished downloads are kept,
# interrupted ones resume. Nothing is installed system-wide and nothing is sent anywhere: the model runs on this Mac only.
# How it runs depends on this Mac's memory: below 48 GB the experts stay on the SSD behind a cache of about 6,600 MiB (smaller for
# contexts above 32k) and 8 of the 10 routed experts are used per token; 48-63 GB: 22,000 MiB cache, all 10 experts;
# 64-127 GB: 35,000 MiB cache, all 10 experts;
# 128 GB and up: the whole model in memory, all 10 experts.
# Options (environment variables): TR_DIR (install folder), TR_PORT (default 8080), TR_CACHE_MIB (expert cache; forces the SSD mode),
# TR_CTX (context window, default 32768; up to 262144), TR_THINK=1 (let the model think before answering; slower), TR_NO_START=1 (download only),
# TR_NGRAM=0 (turn off n-gram drafting, which is on by default: it speeds up answers that repeat text from the prompt, such as code edits).
set -euo pipefail
{ # the whole script is one block, so a cut-off download of this file runs nothing

DIR=${TR_DIR:-$HOME/qwen38-tr}
PORT=${TR_PORT:-8080}
CTX=${TR_CTX:-32768}
REL=llama.cpp-tr-b11721-r7
BIN_URL=https://github.com/ternii-ai/tritq-vllm/releases/download/$REL/llama.cpp-tr-macos-arm64.tar.gz
BIN_SHA=1e4539c60644f0fb75870945280505886768663303994a6ad3ed642567d4c582
HF=https://huggingface.co/Ternii/Qwen3.8-Flash-Next-TR2.31-G-GGUF/resolve/main
MODEL=Qwen3.8-Flash-Next-TR2.31-G
# name  size  sha256
FILES="
Qwen3.8-Flash-Next-TR2.31-G-00001-of-00008.gguf 2276151520 c05ce4861f626550581bc45d587a7f23b3e4d1124d3a3466081e6e756938f124
Qwen3.8-Flash-Next-TR2.31-G-00002-of-00008.gguf 35200169152 3a542dae48de68e443f2830cd081148a8298498bce178d614f33ae216e4a9022
Qwen3.8-Flash-Next-TR2.31-G-00003-of-00008.gguf 9712996704 f2b7e7dbed9990754ad980f6d63a948e2a29b5d5f902e0ca257c2762c833a8c5
Qwen3.8-Flash-Next-TR2.31-G-00004-of-00008.gguf 9724274016 55ccb38251247de3ba3e59a514ff65147928b32cb422dacd56504c196f0d55e8
Qwen3.8-Flash-Next-TR2.31-G-00005-of-00008.gguf 9719729728 82ed8c32ca773b3e0da3304193a6dca15836435bf48d456bfccc679119906eb7
Qwen3.8-Flash-Next-TR2.31-G-00006-of-00008.gguf 9665151104 5e33924c0f22e192dc1168bb96e25878517cda244b78117e3f1542cd25a73c91
Qwen3.8-Flash-Next-TR2.31-G-00007-of-00008.gguf 9687572416 29236756f165e5159f2443b5ecab2af1d5fc7292babb19626135da2b1615a740
Qwen3.8-Flash-Next-TR2.31-G-00008-of-00008.gguf 3791556448 84a0cdc9a06010aedacd2801c1183e65ac128e85d10e09623ef31f577e0290bf
Qwen3.8-Flash-Next-TR2.31-G.trpw 120680652 79cc1c8258622362688196b1832bd352339d4c3b51e152126888a23aaa866ddf
"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31mError: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- checks ----
[ "$(uname -s)" = Darwin ] || die "this script is for macOS."
[ "$(uname -m)" = arm64 ] || die "this needs an Apple silicon Mac (M1 or later)."
OSV=$(sw_vers -productVersion); OSMAJ=${OSV%%.*}; OSMIN=$(echo "$OSV" | cut -d. -f2)
{ [ "$OSMAJ" -gt 13 ] || { [ "$OSMAJ" -eq 13 ] && [ "${OSMIN:-0}" -ge 3 ]; }; } || die "macOS 13.3 or later is needed (this Mac: $OSV)."
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$RAM_GB" -ge 16 ] || die "this model needs a Mac with at least 16 GB of memory (this Mac: $RAM_GB GB)."
mkdir -p "$DIR/model"
NEED=0
while read -r n s h; do
  [ -n "$n" ] || continue
  [ -f "$DIR/model/$n.ok" ] && continue
  have=$(stat -f %z "$DIR/model/$n.part" 2>/dev/null || echo 0); NEED=$(( NEED + s - have ))
done <<< "$FILES"
FREE=$(( $(df -k "$DIR" | awk 'NR==2 {print $4}') * 1024 ))
[ "$FREE" -gt $(( NEED + 2000000000 )) ] || die "not enough free disk space: need about $(( (NEED + 2000000000) / 1000000000 )) GB in $DIR, have $(( FREE / 1000000000 )) GB."

sha_ok() { [ "$(shasum -a 256 "$1" | awk '{print $1}')" = "$2" ]; }

# ---- llama.cpp-tr ----
if [ ! -f "$DIR/$REL.ok" ]; then
  say "Downloading llama.cpp-tr ($REL)..."
  curl -fL --retry 5 --retry-delay 5 -o "$DIR/llama.cpp-tr.tar.gz" "$BIN_URL"
  sha_ok "$DIR/llama.cpp-tr.tar.gz" "$BIN_SHA" || die "llama.cpp-tr download failed its checksum; run the line again."
  rm -rf "$DIR/llama.cpp-tr-macos-arm64"; tar xzf "$DIR/llama.cpp-tr.tar.gz" -C "$DIR"; rm -f "$DIR/llama.cpp-tr.tar.gz"
  xattr -dr com.apple.quarantine "$DIR/llama.cpp-tr-macos-arm64" 2>/dev/null || true
  rm -f "$DIR"/llama.cpp-tr-*.ok; touch "$DIR/$REL.ok"
fi

# ---- model ----
i=0; T=$(echo "$FILES" | grep -c .)
while read -r n s h; do
  [ -n "$n" ] || continue; i=$((i + 1))
  f="$DIR/model/$n"
  [ -f "$f.ok" ] && [ -f "$f" ] && continue
  say "Downloading model file $i of $T: $n ($(( s / 1000000000 )) GB)..."
  [ "$(stat -f %z "$f.part" 2>/dev/null || echo 0)" -gt "$s" ] && rm -f "$f.part"
  if [ "$(stat -f %z "$f.part" 2>/dev/null || echo 0)" != "$s" ]; then
    until curl -fL --retry 10 --retry-delay 10 -C - -o "$f.part" "$HF/$n?download=true"; do
      say "Download interrupted; retrying in 20 s (Ctrl-C to stop; running the line again resumes)."; sleep 20
    done
  fi
  [ "$(stat -f %z "$f.part")" = "$s" ] || { rm -f "$f.part"; die "$n has the wrong size; run the line again."; }
  printf 'Checking %s...\n' "$n"
  sha_ok "$f.part" "$h" || { rm -f "$f.part"; die "$n failed its checksum and was removed; run the line again."; }
  mv "$f.part" "$f"; touch "$f.ok"
done <<< "$FILES"

[ -n "${TR_NO_START:-}" ] && { say "Download complete: $DIR"; exit 0; }

# ---- start ----
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then die "port $PORT is in use (another server running?). Stop it, or set TR_PORT."; fi
RS=(--reasoning off); [ -n "${TR_THINK:-}" ] && RS=()
# n-gram drafting: guesses the next tokens from text already in the conversation and has the model check them in one batch
# (measured: about 2x on code edits with the whole model in memory, +19 % on a 16 GB MacBook Air, no change on new text)
NG=(--spec-type ngram-mod); [ "${TR_NGRAM:-1}" = 0 ] && NG=()
# memory plan (measured on Apple silicon; wired memory in brackets): <48 GB 6,600 MiB cache at 32k, 8 experts (13.4 GB);
# 48-63 GB 22,000 MiB, 10 experts (32 GB); 64-127 GB 35,000 MiB, 10 experts (46 GB); 128 GB+ whole model in memory, 10 experts
# prefill in batches of 2,048 tokens on 48 GB and up (measured: long prompts read +23-36 % faster in memory, +14-74 % in the cache modes);
# below 48 GB too (measured on a 16 GB MacBook Air: +69 % at 3k tokens, +75 % at 9-23k, same answers as batches of 512), with the
# expert cache made smaller as the context grows, so that every layer stays on the GPU (otherwise llama.cpp moves one to the CPU,
# which changes answers and slows decoding); above a 128k context batches stay at 512
PF=(-ub 2048 -b 2048); PE=(GGML_METAL_TQD_MM=1); MB=2048
if [ "$RAM_GB" -ge 128 ] && [ -z "${TR_CACHE_MIB:-}" ]; then
  MODE=(-fit off)
  say "Starting the model fully in memory ($RAM_GB GB Mac, all 10 experts). Loading reads about 90 GB from the SSD the first time."
else
  if [ "$RAM_GB" -ge 64 ]; then CACHE=35000; XP=(LLAMA_MOE_EXPERTS=0)
  elif [ "$RAM_GB" -ge 48 ]; then CACHE=22000; XP=(LLAMA_MOE_EXPERTS=0)
  else XP=()
    # 16 GB plan (MiB): GPU memory = 3,059 model + cache + context + prefill buffers, kept within 10,987 (llama.cpp's own fit target)
    if   [ "$CTX" -le 4096 ];   then CACHE=6700
    elif [ "$CTX" -le 32768 ];  then CACHE=6600
    elif [ "$CTX" -le 65536 ];  then CACHE=5950
    elif [ "$CTX" -le 131072 ]; then CACHE=4750
    else CACHE=3400; PF=(-ub 512 -b 512); PE=(); MB=512; fi
  fi
  CACHE=${TR_CACHE_MIB:-$CACHE}   # the expert cache must take a whole prefill batch (MB)
  MODE=(); XP+=(LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB="$CACHE" LLAMA_MOE_CACHE_MAX_BATCH=$MB LLAMA_MOE_ASYNC=1)
  if [ "$RAM_GB" -ge 48 ]; then say "Starting the model ($RAM_GB GB Mac): experts on the SSD behind a $CACHE MiB cache, all 10 experts."
  else say "Starting the model. Close other large apps first: on a 16 GB Mac it uses about 13 GB."; fi
fi
say "The first answer takes a little longer while macOS reads the model from the SSD. Press Ctrl-C here to stop."
cd "$DIR/llama.cpp-tr-macos-arm64"
env ${XP[@]+"${XP[@]}"} ${PE[@]+"${PE[@]}"} GGML_SCHED_COPY_CB_FIRST=1 GGML_METAL_OPT_GDNPREP=1 GGML_METAL_OPT_NGATED=1 GGML_TQ_NO_REPACK=1 \
  ./llama-server -m "$DIR/model/$MODEL-00001-of-00008.gguf" -c "$CTX" -fa on --jinja -np 1 --cache-ram 0 ${MODE[@]+"${MODE[@]}"} ${RS[@]+"${RS[@]}"} ${NG[@]+"${NG[@]}"} ${PF[@]+"${PF[@]}"} \
  --host 127.0.0.1 --port "$PORT" > "$DIR/server.log" 2>&1 &
SP=$!
# keep the Mac and its GPU awake while the server runs (without this the GPU clocks down and decode is much slower)
caffeinate -ims -w $SP &
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null; echo; say "Stopped."; exit 0' TERM
# Ctrl-C already reaches the server (same process group); a second signal makes it abort mid-cleanup, so signal it only if it is still running after 20 s
trap 'for i in $(seq 40); do kill -0 $SP 2>/dev/null || break; sleep 0.5; done; kill -0 $SP 2>/dev/null && kill $SP 2>/dev/null || true; wait $SP 2>/dev/null || true; echo; say "Stopped."; exit 0' INT
until curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do
  kill -0 $SP 2>/dev/null || { tail -20 "$DIR/server.log"; die "the server stopped while loading (log: $DIR/server.log)."; }
  sleep 2
done
say "Ready: http://127.0.0.1:$PORT (log: $DIR/server.log)"
open "http://127.0.0.1:$PORT" 2>/dev/null || true
wait $SP
}

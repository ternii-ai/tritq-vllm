#!/bin/bash
# SPDX-License-Identifier: MIT
# One-line setup for Qwen3.6-35B-A3B (TR Quantisation, TR2.25-G) on an Apple silicon Mac with 16 GB or more:
#   curl -fsSL https://raw.githubusercontent.com/ternii-ai/tritq-vllm/main/mac/qwen36-35b.sh | bash
# Downloads the ready-built llama.cpp-tr and the model (16.5 GB) into ~/qwen36-35b-tr, checks every file's SHA-256,
# then starts a local chat server and opens it in your browser. Run the same line again to restart; finished downloads are kept,
# interrupted ones resume. Nothing is installed system-wide and nothing is sent anywhere: the model runs on this Mac only.
# On a Mac with less than 24 GB the experts stay on the SSD behind an expert cache; with 24 GB or more the model runs fully in memory.
# Options (environment variables): TR_DIR (install folder), TR_PORT (default 8080), TR_CACHE_MIB (expert cache below 24 GB, default 8400),
# TR_CTX (context window, default 4096), TR_THINK=1 (let the model think before answering; slower), TR_NO_START=1 (download only).
set -euo pipefail
{ # the whole script is one block, so a cut-off download of this file runs nothing

DIR=${TR_DIR:-$HOME/qwen36-35b-tr}
PORT=${TR_PORT:-8080}
CACHE=${TR_CACHE_MIB:-8400}
CTX=${TR_CTX:-4096}
REL=llama.cpp-tr-b11721-r3
BIN_URL=https://github.com/ternii-ai/tritq-vllm/releases/download/$REL/llama.cpp-tr-macos-arm64.tar.gz
BIN_SHA=64f2a57f1b7f7d16c65b30358a21daec0cc019c44ba734913cbcf29ba04777c1
HF=https://huggingface.co/Ternii/Qwen3.6-35B-A3B-TR2.25-G-GGUF/resolve/main
MODEL=Qwen3.6-35B-A3B-TR2.25-G
# name  size  sha256
FILES="
Qwen3.6-35B-A3B-TR2.25-G.gguf 16485874016 47ed879b4109e65d2a387e6547a316b4ea33c5487a1ef33ed18330f257099a1e
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
if [ "$RAM_GB" -lt 24 ]; then
  MOE=(LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB="$CACHE" LLAMA_MOE_CACHE_MAX_BATCH=512 LLAMA_MOE_ASYNC=1 GGML_SCHED_COPY_CB_FIRST=1)
  say "Starting the model with the experts on the SSD ($RAM_GB GB Mac). Close other large apps first: it uses about 12 GB."
else
  MOE=(); say "Starting the model fully in memory ($RAM_GB GB Mac)."
fi
say "The first answer takes a little longer while macOS reads the model from the SSD. Press Ctrl-C here to stop."
cd "$DIR/llama.cpp-tr-macos-arm64"
env ${MOE[@]+"${MOE[@]}"} \
  ./llama-server -m "$DIR/model/$MODEL.gguf" -c "$CTX" -fa on --jinja -np 1 ${RS[@]+"${RS[@]}"} \
  --host 127.0.0.1 --port "$PORT" > "$DIR/server.log" 2>&1 &
SP=$!
# keep the Mac and its GPU awake while the server runs (without this the GPU clocks down and decode is much slower)
caffeinate -ims -w $SP &
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null; echo; say "Stopped."; exit 0' INT TERM
until curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do
  kill -0 $SP 2>/dev/null || { tail -20 "$DIR/server.log"; die "the server stopped while loading (log: $DIR/server.log)."; }
  sleep 2
done
say "Ready: http://127.0.0.1:$PORT (log: $DIR/server.log)"
open "http://127.0.0.1:$PORT" 2>/dev/null || true
wait $SP
}

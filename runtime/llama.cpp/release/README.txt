llama.cpp with the TR Quantisation dense trit types (TQ9_128 / TQ27_128 / TQ81_128)

Ready-built binaries of upstream llama.cpp ff5a233 + patches 0001-0006 from
github.com/ternii-ai/tritq-vllm (runtime/llama.cpp). They load the GGUF files of the TR models
(e.g. huggingface.co/Ternii/Qwen3.6-35B-A3B-TR2.25-G-GGUF), which stock llama.cpp, Ollama and LM Studio cannot.

  macOS arm64 (Apple silicon, macOS 13.3+): Metal; CPU code picked at run time for M1 / M2-M3 / M4 and later.
  Linux arm64 (glibc 2.39+, e.g. Ubuntu 24.04, DGX OS): CPU only; CPU code picked at run time
    (armv8.0 ... armv9.2 + SME). No CUDA: the TQ types are not supported on CUDA by this series.

Quick start
  ./llama-server -m Qwen3.6-35B-A3B-TR2.25-G.gguf -c 4096 -fa on --jinja --reasoning off
  then open http://127.0.0.1:8080 (web UI) or use the OpenAI-compatible API at /v1/chat/completions.

macOS: if the archive was downloaded with a web browser, remove the quarantine flag first:
  xattr -dr com.apple.quarantine llama.cpp-tr-macos-arm64

The expert-cache settings for small-memory Macs, environment switches and known limitations are in
runtime/llama.cpp/README.md of github.com/ternii-ai/tritq-vllm.

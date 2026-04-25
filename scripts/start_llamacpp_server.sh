#!/usr/bin/env bash
# Launch llama-server with project defaults.
# Usage:
#   bash scripts/start_llamacpp_server.sh <model.gguf> [lora.gguf]
#
# Env overrides:
#   LLAMA_CPP_HOME  - required, root of a built llama.cpp clone
#   PORT            - HTTP port (default 8080)
#   CTX_SIZE        - context length (default 16384, matches vLLM max_model_len)
#   PARALLEL        - concurrent slots (default 16)

set -euo pipefail

if [[ -z "${LLAMA_CPP_HOME:-}" ]]; then
    echo "Set LLAMA_CPP_HOME to a built llama.cpp clone." >&2
    exit 1
fi

MODEL="${1:-models/gguf/Qwen3.5-9B.Q4_K_M.gguf}"
LORA="${2:-}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-16384}"
PARALLEL="${PARALLEL:-16}"

SERVER_BIN="$LLAMA_CPP_HOME/build/bin/llama-server"
if [[ ! -x "$SERVER_BIN" ]]; then
    # Windows layout
    if [[ -x "$LLAMA_CPP_HOME/build/bin/Release/llama-server.exe" ]]; then
        SERVER_BIN="$LLAMA_CPP_HOME/build/bin/Release/llama-server.exe"
    else
        echo "llama-server not found at $SERVER_BIN" >&2
        exit 1
    fi
fi

ARGS=(
    -m "$MODEL"
    --ctx-size "$CTX_SIZE"
    --parallel "$PARALLEL"
    --n-gpu-layers -1
    -fa on
    --cont-batching
    --port "$PORT"
    --api-key EMPTY
)

if [[ -n "$LORA" ]]; then
    ARGS+=(--lora "$LORA")
fi

exec "$SERVER_BIN" "${ARGS[@]}"

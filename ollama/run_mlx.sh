#!/bin/sh
set -eu

# Installed alongside the checkpoint in the Mac mini's Ollama storage folder.
MODEL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VIOLETTO_PYTHON=${VIOLETTO_PYTHON:-/Users/marco/violetto-mlx-20260924/.venv/bin/python}
exec "$VIOLETTO_PYTHON" -m mlx_lm generate \
  --model "$MODEL_DIR" --max-tokens 3000 --temp 0.6 --top-p 0.95 "$@"

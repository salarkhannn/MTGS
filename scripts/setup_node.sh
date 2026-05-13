#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}
CUDA_FLAVOR=${CUDA_FLAVOR:-cu121}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_FLAVOR}}

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
pip install transformers datasets accelerate psutil

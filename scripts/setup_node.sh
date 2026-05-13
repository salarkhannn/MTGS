#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}
CUDA_FLAVOR=${CUDA_FLAVOR:-cu121}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_FLAVOR}}
TORCH_VERSION=${TORCH_VERSION:-2.2.2}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.17.2}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.2.2}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-4.40.2}
DATASETS_VERSION=${DATASETS_VERSION:-2.19.1}
ACCELERATE_VERSION=${ACCELERATE_VERSION:-0.30.1}
PSUTIL_VERSION=${PSUTIL_VERSION:-5.9.8}
RUN_VALIDATION=${RUN_VALIDATION:-1}
RUN_MANIFEST=${RUN_MANIFEST:-1}
MANIFEST_PATH=${MANIFEST_PATH:-env_manifest.txt}

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "$TORCH_INDEX_URL"
pip install \
  "transformers==${TRANSFORMERS_VERSION}" \
  "datasets==${DATASETS_VERSION}" \
  "accelerate==${ACCELERATE_VERSION}" \
  "psutil==${PSUTIL_VERSION}"

if [ "$RUN_VALIDATION" = "1" ]; then
  "$PYTHON_BIN" -c "import torch, transformers, datasets, accelerate, psutil; print('torch', torch.__version__); print('transformers', transformers.__version__); print('datasets', datasets.__version__); print('accelerate', accelerate.__version__); print('psutil', psutil.__version__)"
  "$PYTHON_BIN" -c "import torch; print('cuda_available', torch.cuda.is_available())"
fi

if [ "$RUN_MANIFEST" = "1" ]; then
  pip freeze > "$MANIFEST_PATH"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/vllm-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3.5-35B-A3B}"
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-95c0f928cdeeaa21c4906e73cee6a156e1b3b995}"
APT_PACKAGES=(
  build-essential
  ca-certificates
  ccache
  curl
  git
  jq
  pkg-config
  python3
  python3-pip
  python3-venv
  ripgrep
)

run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if [[ ! -d "$ROOT_DIR/vllm/.git" ]]; then
  echo
  echo "Cloning vLLM into $ROOT_DIR/vllm..."
  git clone "$VLLM_REPO_URL" "$ROOT_DIR/vllm"
  git -C "$ROOT_DIR/vllm" checkout "$VLLM_COMMIT"
fi

if ! need_cmd nvidia-smi; then
  echo "nvidia-smi not found. Install the NVIDIA driver first, then re-run setup.sh." >&2
  exit 1
fi

echo "Detected GPU(s):"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if need_cmd apt-get; then
  echo
  echo "Installing system packages..."
  run_as_root apt-get update
  run_as_root apt-get install -y "${APT_PACKAGES[@]}"
fi

if ! "$PYTHON_BIN" - <<'PY'; then
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  echo "$PYTHON_BIN must be Python 3.10 or newer." >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
export MODEL_ID MODEL_DIR

if ! need_cmd uv; then
  echo
  echo "Installing uv..."
  "$PYTHON_BIN" -m pip install --user -U uv
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo
  echo "Creating virtual environment at $VENV_DIR..."
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"

echo
echo "Installing Python packages..."
uv pip install --python "$VENV_PY" -U pip setuptools wheel
uv pip install --python "$VENV_PY" -U \
  "huggingface-hub[cli,hf_transfer]" \
  aiperf \
  nvidia-ml-py

if ! need_cmd claude; then
  echo
  echo "Installing Claude Code..."
  curl -fsSL https://claude.ai/install.sh | bash
fi

echo
echo "Installing vendored vLLM in editable Python-only mode..."
pushd "$ROOT_DIR/vllm" >/dev/null
if ! VLLM_USE_PRECOMPILED=1 uv pip install --python "$VENV_PY" --editable .; then
  echo "Editable install using merge-base wheel failed; retrying against nightly wheels..."
  VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=nightly \
    uv pip install --python "$VENV_PY" --editable .
fi
popd >/dev/null

echo
echo "Downloading model $MODEL_ID into $MODEL_DIR..."
HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV_PY" - <<PY
from pathlib import Path
import os

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
model_dir = Path(os.environ["MODEL_DIR"])
model_dir.parent.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=str(model_dir),
    token=os.environ.get("HF_TOKEN"),
    allow_patterns=[
        "*.json",
        "*.model",
        "*.py",
        "*.safetensors",
        "*.txt",
        "*.tiktoken",
        "*.jinja",
        "LICENSE*",
        "README*",
        "merges.txt",
        "tokenizer*",
        "vocab*",
    ],
    ignore_patterns=[
        "*.bin",
        "*.msgpack",
        "*.onnx",
    ],
)
PY

echo
echo "Setup complete."
echo
echo "Next steps:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  claude"
echo "  python benchmark.py"
echo
echo "Model downloaded to:"
echo "  $MODEL_DIR"

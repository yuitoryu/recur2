#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

uv_bin="${UV_BIN:-uv}"
if ! command -v "$uv_bin" >/dev/null && [[ -x ~/.local/bin/uv ]]; then
    uv_bin=~/.local/bin/uv
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PWD/.cache/python}"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"

factory=third_party/LLaMA-Factory
if [[ ! -f "$factory/pyproject.toml" ]]; then
    git submodule update --init -- "$factory"
fi
if [[ ! -x .venv/bin/python ]]; then
    "$uv_bin" venv --python "${RECUR_PYTHON:-3.11}" .venv
fi

requirements=requirements-wsl.txt
if [[ -f requirements-wsl.lock ]]; then
    requirements=requirements-wsl.lock
fi

"$uv_bin" pip install --python .venv/bin/python \
    --index https://download.pytorch.org/whl/cu128 \
    'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' 'torchaudio==2.8.0+cu128'
"$uv_bin" pip install --python .venv/bin/python \
    --index https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-first-match \
    --constraint "$requirements" --editable "$factory"
if [[ ${1:-} == --fa2 ]]; then
    "$uv_bin" pip install --python .venv/bin/python --no-deps --requirements requirements-fa2-wsl.txt
fi
"$uv_bin" pip check --python .venv/bin/python
.venv/bin/llamafactory-cli version

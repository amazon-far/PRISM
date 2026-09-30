#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash install.sh"
  echo "Install simulation dependencies into the active Python 3.11 environment."
  echo "PRISM_PYTHON selects the interpreter; PRISM_ISAACLAB_ROOT overrides the source cache."
  exit 0
fi
if (($#)); then echo "Unknown argument: $1" >&2; exit 2; fi
cd "$ROOT_DIR"
python_bin="${PRISM_PYTHON:-python3}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
"$python_bin" "$ROOT_DIR/scripts/check_environment.py" --platform-only
"$python_bin" -m pip install 'uv==0.11.8'

isaaclab_commit=3c6e67bb5c7ada942a6d1884ab69338f57596f77
isaaclab_root="${PRISM_ISAACLAB_ROOT:-$("$python_bin" -c 'import sys; print(sys.prefix)')/share/prism/IsaacLab}"
if [[ ! -e "$isaaclab_root" ]]; then
  git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$isaaclab_root"
fi
if [[ "$(git -C "$isaaclab_root" rev-parse HEAD)" != "$isaaclab_commit" ]] ||
   [[ -n "$(git -C "$isaaclab_root" status --porcelain)" ]]; then
  echo "Isaac Lab must be a clean checkout of $isaaclab_commit: $isaaclab_root" >&2
  exit 1
fi
# CUDA wheels come from PyTorch's index, independently of the host CUDA toolkit.
"$python_bin" -m uv --no-config pip install --python "$python_bin" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128'
"$python_bin" -m uv --no-config pip install --python "$python_bin" \
  --default-index https://pypi.org/simple --index https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  --build-constraints "$ROOT_DIR/requirements/build-constraints.txt" \
  --overrides "$ROOT_DIR/requirements/simulation-overrides.txt" \
  -r "$ROOT_DIR/requirements/simulation.txt" \
  -c "$ROOT_DIR/requirements/simulation-lock.txt" -e "$isaaclab_root/source/isaaclab"
"$python_bin" "$ROOT_DIR/scripts/check_environment.py"

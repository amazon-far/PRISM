#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
python_bin="${PRISM_PYTHON:-python3}"
mode="${1:-ffs}"
case "$mode" in
  --help|-h)
    echo "Usage: bash install.sh [--relay | --native-depth]"
    echo "Default: pinned FFS submodule, CUDA PyTorch, camera dependencies and policy SDK."
    echo "Run inside a Python 3.10 virtual environment; --relay installs only camera dependencies."
    exit 0 ;;
  ffs|--relay|--native-depth) ;;
  *) echo "Unknown option: $mode" >&2; exit 2 ;;
esac
"$python_bin" - "$mode" <<'PY'
import sys, platform
if sys.prefix == sys.base_prefix and not __import__('os').environ.get('CONDA_PREFIX'):
    sys.exit('Activate a virtual environment first (see README.md).')
if sys.argv[1] != '--relay' and sys.version_info[:2] != (3, 10):
    sys.exit('Policy SDK installation requires Python 3.10.')
if sys.argv[1] == 'ffs' and (platform.system() != 'Linux' or platform.machine() != 'x86_64'):
    sys.exit('The bundled CUDA wheels target Linux x86_64. Use --relay on the camera host.')
PY
"$python_bin" -m pip install --upgrade pip setuptools wheel
if [[ "$mode" == "--relay" ]]; then
  "$python_bin" -m pip install -r requirements-relay.txt
  "$python_bin" scripts/check_install.py --relay
elif [[ "$mode" == "--native-depth" ]]; then
  "$python_bin" -m pip install -r requirements.txt
  "$python_bin" scripts/check_install.py --native-depth
else
  git submodule update --init --recursive -- third_party/Fast-FoundationStereo
  "$python_bin" -m pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3 \
    --index-url https://download.pytorch.org/whl/cu124
  "$python_bin" -m pip install -r requirements-ffs.txt
  "$python_bin" scripts/check_install.py
fi
"$python_bin" -m pip check

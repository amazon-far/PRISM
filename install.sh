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
if (($# > 1)); then echo "Unexpected argument: $2" >&2; exit 2; fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
"$python_bin" - "$mode" <<'PY'
import sys, platform
import ctypes
from pathlib import Path
if sys.prefix == sys.base_prefix and not (Path(sys.prefix) / 'conda-meta').is_dir():
    sys.exit('Activate a virtual environment first (see README.md).')
cfg = Path(sys.prefix) / 'pyvenv.cfg'
if cfg.is_file() and 'include-system-site-packages = true' in cfg.read_text().lower():
    sys.exit('Create the virtual environment without --system-site-packages.')
if sys.argv[1] != '--relay' and sys.version_info[:2] != (3, 10):
    sys.exit('Policy SDK installation requires Python 3.10.')
if sys.argv[1] == 'ffs' and (platform.system() != 'Linux' or platform.machine() != 'x86_64'):
    sys.exit('The bundled CUDA wheels target Linux x86_64. Use --relay on the camera host.')
if platform.system() == 'Linux':
    libraries = {'libusb-1.0.so.0': 'libusb-1.0-0'}
    if sys.argv[1] == 'ffs':
        libraries.update({'libGL.so.1': 'libgl1', 'libEGL.so.1': 'libegl1',
                          'libgthread-2.0.so.0': 'libglib2.0-0'})
    missing = []
    for library, package in libraries.items():
        try:
            ctypes.CDLL(library)
        except OSError:
            missing.append(package)
    if missing:
        sys.exit('Missing system libraries. On Ubuntu: sudo apt install ' + ' '.join(missing))
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
  "$python_bin" -m pip install -r requirements-ffs.txt -c constraints-ffs.txt
  "$python_bin" scripts/check_install.py
fi
"$python_bin" -m pip check

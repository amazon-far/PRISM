#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
python_bin="${PRISM_PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash real_depth.sh [image-server options]"
  echo "Streams D435i depth using the original real_d435i configuration."
  exit 0
fi

log_dir="$ROOT_DIR/logs/real_depth_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$log_dir"
if [[ "${HOLOSOMA_DEPLOYMENT_AUDIT:-0}" == "1" ]]; then
  export HOLOSOMA_DEPLOYMENT_AUDIT_DIR="$log_dir/evidence"
else
  unset HOLOSOMA_DEPLOYMENT_AUDIT_DIR
fi
exec > >(tee -a "$log_dir/depth.log") 2>&1
echo "[real_depth] log_dir=$log_dir"
exec "$python_bin" -m holosoma.sensors.image_server real_d435i \
  --image-saver-config.image-root-dir "$log_dir/depth_images" "$@"

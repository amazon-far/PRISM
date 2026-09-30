#!/usr/bin/env bash
# Run the policy on an existing FFS depth stream.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/scripts/ffs_common.sh"
if [[ "${HOLOSOMA_DEPTH_CHECK_ONLY:-0}" != "1" ]]; then ffs_require_policy; fi
cd "$ROOT_DIR"
"$python_bin" scripts/check_depth_stream.py --timeout "${HOLOSOMA_SHM_WAIT_S:-120}"
if [[ "${HOLOSOMA_DEPTH_CHECK_ONLY:-0}" == "1" ]]; then exit 0; fi
export HOLOSOMA_ORT_THREADS="${HOLOSOMA_ORT_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec bash "$ROOT_DIR/real_drop.sh" --model "$HOLOSOMA_REAL_MODEL_PATH" \
  --interface "$HOLOSOMA_REAL_INTERFACE" -- "${policy_args[@]}"

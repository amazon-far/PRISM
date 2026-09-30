#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PRISM_PYTHON:-python3}"
checkpoint="${HOLOSOMA_REAL_MODEL_PATH:-}"
interface="${HOLOSOMA_REAL_INTERFACE:-eth0}"
extra_args=()
while (($#)); do
  case "$1" in
    --model|--interface)
      if (($# < 2)) || [[ -z "$2" ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      if [[ "$1" == "--model" ]]; then checkpoint="$2"; else interface="$2"; fi
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash real_drop.sh --model /path/to/policy.onnx [--interface eth0] [-- policy options]"
      echo "HOLOSOMA_REAL_MODEL_PATH and HOLOSOMA_REAL_INTERFACE also set these options."
      exit 0
      ;;
    --) shift; extra_args=("$@"); break ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 2 ;;
  esac
done
if [[ -z "$checkpoint" || ! -f "$checkpoint" || "$checkpoint" != *.onnx ]]; then
  echo "Provide an existing ONNX model with --model or HOLOSOMA_REAL_MODEL_PATH." >&2
  exit 2
fi
checkpoint="$(realpath -- "$checkpoint")"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src/holosoma_inference:$ROOT_DIR/src/holosoma"

log_dir="$ROOT_DIR/logs/real_drop_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$log_dir"
if [[ "${HOLOSOMA_DEPLOYMENT_AUDIT:-0}" == "1" ]]; then
  export HOLOSOMA_DEPLOYMENT_AUDIT_DIR="$log_dir/evidence"
else
  unset HOLOSOMA_DEPLOYMENT_AUDIT_DIR
fi
exec > >(tee -a "$log_dir/run.log") 2>&1
echo "[real_drop] checkpoint=$checkpoint"
echo "[real_drop] interface=$interface"
"$python_bin" "$ROOT_DIR/scripts/show_policy_command.py" "$log_dir/latest_command.json" &
command_window_pid=$!
trap 'kill "$command_window_pid" 2>/dev/null || true' EXIT

HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=0 \
HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}" \
HOLOSOMA_POLICY_COMMAND_STATUS_PATH="$log_dir/latest_command.json" \
HOLOSOMA_POLICY_DEBUG_INPUT_PATH="$log_dir/depth_command.jsonl" \
HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="${HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT:-100000}" \
"$python_bin" -m holosoma_inference.run_policy \
  inference:g1-root_pos-contact-aware-drop-button-actions-no-linvel-h1 \
  --task.model-path "$checkpoint" \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface "$interface" "${extra_args[@]}"

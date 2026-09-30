#!/usr/bin/env bash
# Shared argument handling for the FFS launchers. Sourced from the repository root.
python_bin="${PRISM_PYTHON:-${HOLOSOMA_INFERENCE_PYTHON:-python3}}"
export PRISM_PYTHON="$python_bin"
export HOLOSOMA_FFS_REPO="${HOLOSOMA_FFS_REPO:-$ROOT_DIR/third_party/Fast-FoundationStereo}"
export HOLOSOMA_REAL_INTERFACE="${HOLOSOMA_REAL_INTERFACE:-eth0}"
export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference"
policy_args=()
while (($#)); do
  case "$1" in
    --model|--ffs-model|--relay-host|--connect|--interface|--relay-python)
      if (($# < 2)) || [[ -z "$2" ]]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --model) export HOLOSOMA_REAL_MODEL_PATH="$2" ;;
        --ffs-model) export HOLOSOMA_FFS_MODEL="$2" ;;
        --relay-host) export HOLOSOMA_RELAY_HOST="$2" ;;
        --connect) export HOLOSOMA_REMOTE_STEREO_CONNECT="$2" HOLOSOMA_RELAY_HOST="" ;;
        --interface) export HOLOSOMA_REAL_INTERFACE="$2" ;;
        --relay-python) export HOLOSOMA_RELAY_PYTHON="$2" ;;
      esac
      shift 2 ;;
    --help|-h)
      echo "Usage: bash $(basename "${BASH_SOURCE[1]}") [options] [-- policy options]"
      echo "  --model FILE          Student ONNX policy (policy launchers)"
      echo "  --ffs-model FILE      FFS weights; cfg.yaml must be in the same directory"
      echo "  --relay-host HOST     Start and stop the camera relay through SSH"
      echo "  --connect tcp://HOST:5602   Use a relay already running at this address"
      echo "  --relay-python PATH   Python on the camera host (default: python3)"
      echo "  --interface NIC       Robot control interface (default: eth0)"
      exit 0 ;;
    --) shift; policy_args=("$@"); break ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 2 ;;
  esac
done
# Resolve model paths relative to the caller before any launcher changes directory.
for name in HOLOSOMA_REAL_MODEL_PATH HOLOSOMA_FFS_MODEL; do
  value="${!name:-}"
  if [[ -n "$value" && -f "$value" ]]; then export "$name=$(realpath -- "$value")"; fi
done
ffs_require_policy() {
  if [[ -z "${HOLOSOMA_REAL_MODEL_PATH:-}" || ! -f "$HOLOSOMA_REAL_MODEL_PATH" || "$HOLOSOMA_REAL_MODEL_PATH" != *.onnx ]]; then
    echo "Provide an existing student ONNX with --model or HOLOSOMA_REAL_MODEL_PATH." >&2; return 2
  fi
}
ffs_require_depth() {
  if [[ ! -f "$HOLOSOMA_FFS_REPO/core/foundation_stereo.py" ]]; then
    echo "FFS submodule is missing; run bash install.sh." >&2; return 2
  fi
  if [[ -z "${HOLOSOMA_FFS_MODEL:-}" || ! -f "$HOLOSOMA_FFS_MODEL" || ! -f "$(dirname "$HOLOSOMA_FFS_MODEL")/cfg.yaml" ]]; then
    echo "Provide FFS weights with --ffs-model and cfg.yaml in the same directory." >&2; return 2
  fi
  if [[ -z "${HOLOSOMA_RELAY_HOST:-}" && -z "${HOLOSOMA_REMOTE_STEREO_CONNECT:-}" ]]; then
    echo "Set --relay-host for SSH management or --connect for an existing stereo relay." >&2; return 2
  fi
}

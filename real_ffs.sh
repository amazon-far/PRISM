#!/usr/bin/env bash
# Start the stereo relay, FFS depth server and policy together.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/scripts/ffs_common.sh"
ffs_require_depth
if [[ "${HOLOSOMA_DRY_RUN:-0}" != "1" ]]; then ffs_require_policy; fi
cd "$ROOT_DIR"
if pgrep -f '[h]olosoma.sensors.image_server|[i]mage_server.py' >/dev/null; then
  echo "An image server is already running; stop it or use real_ffs_run.sh with that stream." >&2
  exit 1
fi
session_dir="$ROOT_DIR/logs/real_ffs_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$session_dir"
depth_pid=""
policy_pid=""
cleanup() {
  trap - EXIT INT TERM
  for pid in "$policy_pid" "$depth_pid"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      for _ in {1..40}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
      kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid bash "$ROOT_DIR/real_ffs_depth.sh" > "$session_dir/depth.log" 2>&1 < /dev/null &
depth_pid=$!
echo "FFS depth starting; log: $session_dir/depth.log"
if [[ "${HOLOSOMA_DRY_RUN:-0}" == "1" ]]; then
  wait "$depth_pid"
else
  # real_ffs_run.sh performs the source's depth checks before starting the policy.
  setsid bash "$ROOT_DIR/real_ffs_run.sh" -- "${policy_args[@]}" <&0 &
  policy_pid=$!
  while kill -0 "$policy_pid" 2>/dev/null; do
    if ! kill -0 "$depth_pid" 2>/dev/null; then
      echo "Depth server exited; stopping the policy. See $session_dir/depth.log" >&2
      exit 1
    fi
    sleep 0.2
  done
  wait "$policy_pid"
fi

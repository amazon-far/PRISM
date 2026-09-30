#!/usr/bin/env bash
# Receive remote IR stereo and publish FFS depth for the policy.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/scripts/ffs_common.sh"
ffs_require_depth
cd "$ROOT_DIR"
export HOLOSOMA_DEPTH_PREDICTOR=ffs
RELAY_HOST="${HOLOSOMA_RELAY_HOST:-}"
RELAY_PORT="${HOLOSOMA_RELAY_PORT:-5602}"
RELAY_DIR="${HOLOSOMA_RELAY_REMOTE_DIR:-.cache/prism-relay}"
RELAY_PYTHON="${HOLOSOMA_RELAY_PYTHON:-python3}"
RELAY_STOP_SERVICES="${HOLOSOMA_RELAY_STOP_SERVICES:-}"
# These fields enter remote shell commands; accept simple paths/service names only.
[[ "$RELAY_DIR" =~ ^[a-zA-Z0-9_./-]+$ && "$RELAY_DIR" != -* ]] || { echo "Invalid relay directory" >&2; exit 2; }
[[ "$RELAY_PYTHON" =~ ^[a-zA-Z0-9_./-]+$ && "$RELAY_PYTHON" != -* ]] || { echo "Invalid relay Python path" >&2; exit 2; }
[[ "$RELAY_PORT" =~ ^[0-9]+$ ]] && ((RELAY_PORT > 0 && RELAY_PORT < 65536)) || { echo "Invalid relay port" >&2; exit 2; }
[[ "$RELAY_STOP_SERVICES" =~ ^[a-zA-Z0-9_.@\ -]*$ ]] || { echo "Invalid service names" >&2; exit 2; }
if [[ -n "$RELAY_HOST" ]]; then
  [[ "$RELAY_HOST" =~ ^[a-zA-Z0-9_@.-]+$ && "$RELAY_HOST" != -* ]] || { echo "Invalid SSH host" >&2; exit 2; }
fi
export HOLOSOMA_REMOTE_STEREO_CONNECT="${HOLOSOMA_REMOTE_STEREO_CONNECT:-tcp://${RELAY_HOST##*@}:${RELAY_PORT}}"

log_dir="${ROOT_DIR}/logs/real_ffs_depth_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$log_dir"

# Record what the policy actually saw. Each .npz under evidence/ holds the metric
# FFS depth, the crop/resize/clip result, the delayed 58x87 frame written to shared
# memory, and the IR stereo pair it came from. Measured at ~1.4 MB per record, so
# every 3rd frame (10 Hz, ~14 MB/s, ~50 GB/h) by default; EVERY=1 keeps every
# frame but the compressor cannot keep up at 30 Hz and starts dropping records.
# HOLOSOMA_DEPLOYMENT_AUDIT=1 enables recording.
if [[ "${HOLOSOMA_DEPLOYMENT_AUDIT:-0}" == "1" ]]; then
  export HOLOSOMA_DEPLOYMENT_AUDIT_DIR="${log_dir}/evidence"
  export HOLOSOMA_AUDIT_DEPTH_EVERY="${HOLOSOMA_AUDIT_DEPTH_EVERY:-3}"
  export HOLOSOMA_AUDIT_DEPTH_LIMIT="${HOLOSOMA_AUDIT_DEPTH_LIMIT:-36000}"
else
  unset HOLOSOMA_DEPLOYMENT_AUDIT_DIR
fi

exec > >(tee -a "${log_dir}/depth.log") 2>&1
echo "[real_ffs_depth] log_dir=${log_dir}"
[[ -n "${HOLOSOMA_DEPLOYMENT_AUDIT_DIR:-}" ]] && echo "[real_ffs_depth] recording depth evidence to ${HOLOSOMA_DEPLOYMENT_AUDIT_DIR} (every ${HOLOSOMA_AUDIT_DEPTH_EVERY} frame(s), limit ${HOLOSOMA_AUDIT_DEPTH_LIMIT})"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5)

stop_remote_relay() {
  [[ "${relay_started:-0}" == "1" ]] || return 0
  local out
  out=$("${SSH[@]}" "$RELAY_HOST" "if [ -f ${RELAY_DIR}/relay.pid ]; then kill \"\$(cat ${RELAY_DIR}/relay.pid)\" 2>/dev/null && echo killed; rm -f ${RELAY_DIR}/relay.pid; fi
    if [ -f ${RELAY_DIR}/services.stopped ]; then for s in \$(cat ${RELAY_DIR}/services.stopped); do systemctl --user start \"\$s\" && echo \"restarted \$s\"; done; rm -f ${RELAY_DIR}/services.stopped; fi" 2>/dev/null) \
    || { echo "[real_ffs_depth] WARNING: could not reach ${RELAY_HOST} to stop the relay; check ${RELAY_DIR}/relay.pid there" >&2; return 0; }
  [[ "$out" == *killed* ]] && echo "[real_ffs_depth] remote relay stopped"
  [[ "$out" == *restarted* ]] && echo "[real_ffs_depth] remote services restored: $(printf '%s' "$out" | sed -n 's/^restarted //p' | tr '\n' ' ')"
  return 0
}

install_remote_relay() {
  # The relay is two files that live in this repo. Push them to the robot when
  # they are missing or differ, so a wiped home directory or a stale copy on the
  # Jetson can never be the reason deployment fails.
  local remote_sums
  remote_sums=$("${SSH[@]}" "$RELAY_HOST" "mkdir -p ${RELAY_DIR} && cd ${RELAY_DIR} && md5sum stereo_relay_pub.py real_ffs_relay.sh 2>/dev/null | awk '{print \$1}' | tr '\n' ' '") \
    || { echo "[real_ffs_depth] ERROR: cannot ssh to ${RELAY_HOST}" >&2; return 1; }
  local local_sums
  local_sums=$(md5sum scripts/stereo_relay_pub.py scripts/real_ffs_relay.sh | awk '{print $1}' | tr '\n' ' ')
  if [[ "$remote_sums" != "$local_sums" ]]; then
    echo "[real_ffs_depth] installing relay scripts to ${RELAY_HOST}:${RELAY_DIR}"
    scp -q -o BatchMode=yes -o ConnectTimeout=5 \
      scripts/stereo_relay_pub.py scripts/real_ffs_relay.sh requirements-relay.txt "${RELAY_HOST}:${RELAY_DIR}/" \
      || { echo "[real_ffs_depth] ERROR: scp to ${RELAY_HOST} failed" >&2; return 1; }
    "${SSH[@]}" "$RELAY_HOST" "chmod +x ${RELAY_DIR}/real_ffs_relay.sh"
  fi
}

start_remote_relay() {
  install_remote_relay || exit 1
  echo "[real_ffs_depth] starting stereo relay on ${RELAY_HOST} (${RELAY_DIR}/real_ffs_relay.sh)"
  # Kill a previous relay via its pidfile, launch detached from this ssh session,
  # then block until it reports it is publishing so image_server never has to wait.
  "${SSH[@]}" "$RELAY_HOST" "bash -s" <<EOF
set -e
cd ${RELAY_DIR}
[ -f relay.pid ] && kill "\$(cat relay.pid)" 2>/dev/null && sleep 0.5
rm -f relay.log services.stopped
for s in ${RELAY_STOP_SERVICES}; do
  if systemctl --user is-active --quiet "\$s"; then
    systemctl --user stop "\$s" && echo "\$s" >> services.stopped && echo "  stopped \$s (will restart on exit)"
  fi
done
[ -s services.stopped ] && sleep 1.5
HOLOSOMA_RELAY_PYTHON='${RELAY_PYTHON}' HOLOSOMA_RELAY_BIND='tcp://*:${RELAY_PORT}' setsid nohup ./real_ffs_relay.sh > relay.log 2>&1 < /dev/null &
echo \$! > relay.pid
for i in \$(seq 1 480); do
  grep -q publishing relay.log 2>/dev/null && { echo "  relay pid \$(cat relay.pid): \$(grep -m1 publishing relay.log)"; exit 0; }
  kill -0 "\$(cat relay.pid)" 2>/dev/null || break
  sleep 0.25
done
echo "  relay did not start within 120 s; remote log:" >&2; tail -8 relay.log >&2
echo "  camera holders now:" >&2; for v in /dev/video*; do fuser "\$v" 2>/dev/null | xargs -r -n1 ps -o pid=,cmd= -p 2>/dev/null; done | sort -u | cut -c1-120 | sed 's/^/    /' >&2
exit 1
EOF
}

image_server_pid=""
status_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$status_pid" ]]; then
    kill "$status_pid" 2>/dev/null || true
    wait "$status_pid" 2>/dev/null || true
  fi
  if [[ -n "$image_server_pid" ]] && kill -0 "$image_server_pid" 2>/dev/null; then
    kill -TERM "$image_server_pid" 2>/dev/null
    wait "$image_server_pid" 2>/dev/null || true
  fi
  stop_remote_relay
}
trap cleanup EXIT
trap 'echo "[real_ffs_depth] stop requested"; cleanup; exit 130' INT TERM

if [[ -n "$RELAY_HOST" ]]; then
  relay_started=1
  start_remote_relay
else
  echo "[real_ffs_depth] HOLOSOMA_RELAY_HOST is empty: assuming the relay is already running at ${HOLOSOMA_REMOTE_STEREO_CONNECT}"
fi

echo "[real_ffs_depth] stereo source=${HOLOSOMA_REMOTE_STEREO_CONNECT}  predictor=ffs  model=${HOLOSOMA_FFS_MODEL}"
# A segment left by a killed server would be reused ("Connected to existing shared
# memory") and briefly serve its last frame; start from a clean one instead.
if [[ -e /dev/shm/depth_img_shm ]] && ! pgrep -f "[h]olosoma.sensors.image_server|[i]mage_server.py" >/dev/null 2>&1; then
  echo "[real_ffs_depth] removing stale depth_img_shm left by a previous session"
  rm -f /dev/shm/depth_img_shm
fi
"$python_bin" -m holosoma.sensors.image_server remote_ffs_d435i \
  --image-saver-config.image-root-dir "${log_dir}/depth_images" &
image_server_pid=$!

# This script is only the depth half and runs until stopped; the image server's
# stats will scroll below. Say so once the segment is live, because it is easy to
# sit here waiting for the policy's prompt that will never come from this process.
(
  for _ in $(seq 1 90); do [[ -e /dev/shm/depth_img_shm ]] && break; sleep 1; done
  if [[ -e /dev/shm/depth_img_shm ]]; then
    echo
    echo "================================================================================"
    echo "[real_ffs_depth] DEPTH IS RUNNING. This terminal only serves depth."
    echo "[real_ffs_depth] To run the policy, open another terminal:  bash real_ffs_run.sh"
    echo "[real_ffs_depth] (or stop this and use the single command:  bash real_ffs.sh)"
    echo "================================================================================"
    echo
  fi
) &
status_pid=$!
wait "$image_server_pid"

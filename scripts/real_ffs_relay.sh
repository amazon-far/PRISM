#!/usr/bin/env bash
# Robot side of the FFS depth path. Runs on the Jetson the D435i is plugged into
# and streams the IR stereo pair + calibration to the laptop, where
# real_ffs_depth.sh runs Fast-FoundationStereo on it. Start this first.
#
# Keep the IR emitter ON: with it off both the D435i's own depth and FFS depth
# get markedly noisier (measured: FFS frame-to-frame jitter 3x, D435i 4x).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${HOLOSOMA_RELAY_PYTHON:-${PRISM_PYTHON:-python3}}"
if ! "$PY" -c "import pyrealsense2, zmq, numpy"; then
  echo "Install camera dependencies first: python3 -m pip install -r requirements-relay.txt" >&2
  exit 1
fi
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/real_ffs_relay.sh (configure HOLOSOMA_RELAY_PYTHON/BIND/EMITTER as needed)"
  exit 0
fi
for tool in lsusb fuser; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; install usbutils and psmisc." >&2; exit 1; }
done
BIND="${HOLOSOMA_RELAY_BIND:-tcp://*:5602}"
EMITTER="${HOLOSOMA_RELAY_EMITTER:-on}"

if [[ -z "$PY" ]]; then
  echo "[real_ffs_relay] ERROR: no python with pyrealsense2 + pyzmq + numpy found (set HOLOSOMA_RELAY_PYTHON)" >&2
  exit 1
fi
# The D435i drops off the bus for a few seconds at a time on this robot and then
# re-enumerates on its own. Wait for it instead of failing the whole deployment.
cam_wait="${HOLOSOMA_RELAY_CAMERA_WAIT_S:-90}"
for ((t = 0; t < cam_wait; t++)); do
  lsusb 2>/dev/null | grep -q "8086:0b3a" && break
  (( t == 0 )) && echo "[real_ffs_relay] D435i (8086:0b3a) not on the USB bus; waiting up to ${cam_wait}s for it to re-enumerate..." >&2
  sleep 1
done
if ! lsusb 2>/dev/null | grep -q "8086:0b3a"; then
  echo "[real_ffs_relay] ERROR: no D435i (8086:0b3a) on the USB bus after ${cam_wait}s. Re-seat the camera cable." >&2
  exit 1
fi
# Give udev a moment to create the video nodes after a re-enumeration.
for ((t = 0; t < 10; t++)); do [[ $(ls /dev/video* 2>/dev/null | wc -l) -ge 6 ]] && break; sleep 0.5; done

# The D435i can only be streamed by one process. Another RealSense client (e.g. an
# autostarted stereo server) makes our pipeline start fine and then starve with
# "Frame didn't arrive", so refuse up front and say who has it rather than fight.
holders=""
for v in /dev/video*; do
  for p in $(fuser "$v" 2>/dev/null); do
    [[ "$p" == "$$" ]] && continue
    holders+="$(ps -o pid=,cmd= -p "$p" 2>/dev/null | cut -c1-110)"$'\n'
  done
done
holders="$(printf '%s' "$holders" | sort -u | sed '/^$/d')"
if [[ -n "$holders" ]]; then
  echo "[real_ffs_relay] ERROR: the camera is already in use by another process:" >&2
  printf '%s\n' "$holders" | sed 's/^/    /' >&2
  echo "[real_ffs_relay] Stop it first (or set HOLOSOMA_RELAY_IGNORE_HOLDERS=1 to try anyway)." >&2
  [[ "${HOLOSOMA_RELAY_IGNORE_HOLDERS:-0}" == "1" ]] || exit 1
fi

echo "[real_ffs_relay] bind=${BIND} emitter=${EMITTER}"
exec "$PY" -u "${HERE}/stereo_relay_pub.py" --bind "$BIND" --emitter "$EMITTER" --stats-interval 5

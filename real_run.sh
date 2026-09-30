#!/usr/bin/env bash
# Compatibility entrypoint for the explicit-model policy launcher.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT_DIR/real_drop.sh" "$@"

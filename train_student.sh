#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${repo_root}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" -m scripts._student "$@"

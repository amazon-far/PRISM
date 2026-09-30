#!/usr/bin/env bash
# Read-only GPU diagnostics; no machine-specific clocks, power changes or load test.
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,pstate,utilization.gpu,memory.used,memory.total,clocks.sm,power.draw,power.limit --format=csv

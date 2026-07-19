#!/usr/bin/env bash
# Monitor era5_n96_2020_2021.zarr build progress (bash/WSL/Git Bash).
# Usage: bash scripts/monitor_zarr_build.sh [interval_seconds]

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZARR="${ROOT}/data/processed/lapai/era5_n96_2020_2021.zarr"
LOG="${ROOT}/reports/zarr_build_debug.log"
if [[ ! -f "${LOG}" ]]; then
  LOG="${ROOT}/reports/zarr_build_stderr.log"
fi
INTERVAL="${1:-60}"

echo "Watching ${ZARR}"
echo "Log: ${LOG} (Ctrl+C to stop)"
while true; do
  chunks=0
  mb="0.0"
  if [[ -d "${ZARR}/data" ]]; then
    chunks="$(find "${ZARR}/data" -maxdepth 1 -type f ! -name '.*' 2>/dev/null | wc -l | tr -d ' ')"
    if command -v du >/dev/null 2>&1; then
      mb="$(du -sm "${ZARR}" 2>/dev/null | awk '{print $1}')"
    fi
  fi
  alive="no"
  if pgrep -f "anemoi-datasets.*era5_n96_2020_2021" >/dev/null 2>&1; then
    alive="yes"
  fi
  tail_line="(no build log)"
  if [[ -f "${LOG}" ]]; then
    tail_line="$(tail -n 1 "${LOG}" 2>/dev/null || echo '(empty log)')"
  fi
  printf '[%s] alive=%s data_chunks=%s size_mb=%s\n' "$(date +%H:%M:%S)" "${alive}" "${chunks}" "${mb}"
  printf '  %s\n' "${tail_line}"
  sleep "${INTERVAL}"
done

#!/usr/bin/env bash
# Resumable upload of the full Track A training Zarr from the laptop to Lengau lustre.
#
# Run from WSL:
#   tr -d '\r' < scripts/upload_zarr_to_lengau.sh > /tmp/up.sh && bash /tmp/up.sh
#
# Lengau is only reachable on a private VPN address; WSL has no split-horizon DNS,
# so we connect by IP and pin HostKeyAlias so the known_hosts entry stays stable.
set -uo pipefail

SRC="/mnt/c/Users/MthethoSovara/tiny-media-analysis/lapai-forecast/data/processed/lapai/era5_n96_2020_2021.zarr/"
DEST_HOST="msovara@10.128.15.127"
DEST_PATH="/home/msovara/lustre/lapai-data/era5_n96_2020_2021.zarr/"
LOG="/mnt/c/Users/MthethoSovara/tiny-media-analysis/lapai-forecast/reports/zarr_upload_lengau.log"
KEY="${HOME}/.ssh/id_ed25519"

SSH_CMD="ssh -i ${KEY} -o HostKeyAlias=lengau.chpc.ac.za -o StrictHostKeyChecking=accept-new"
SSH_CMD="${SSH_CMD} -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3"

if [ ! -f "${KEY}" ]; then
  echo "ERROR: ssh key not found: ${KEY}" >&2
  exit 1
fi
if [ ! -d "${SRC}" ]; then
  echo "ERROR: source Zarr not found: ${SRC}" >&2
  exit 1
fi

# Zarr chunks are already blosc/lz4 compressed, so -z would burn CPU for nothing.
# --partial keeps interrupted chunk files so a re-run resumes instead of restarting.
{
  echo "=== UPLOAD START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "src : ${SRC}"
  echo "dest: ${DEST_HOST}:${DEST_PATH}"
  echo
} >> "${LOG}"

nohup rsync -a --partial --stats --info=progress2,name0 \
  -e "${SSH_CMD}" \
  "${SRC}" "${DEST_HOST}:${DEST_PATH}" \
  >> "${LOG}" 2>&1 &

PID=$!
echo "RSYNC_PID=${PID}"
echo "LOG=${LOG}"
disown "${PID}" 2>/dev/null || true
sleep 5
if kill -0 "${PID}" 2>/dev/null; then
  echo "STATUS=running"
else
  echo "STATUS=exited_early -- check log"
  tail -20 "${LOG}"
fi

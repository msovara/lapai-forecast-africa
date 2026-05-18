#!/usr/bin/env bash
# Shared conda / module bootstrap for LapAI-Forecast PBS jobs on CHPC Lengau.
#
# Each PBS script must first:
#   cd "${PBS_O_WORKDIR}" || exit 1   # submit from lapai-forecast repo root
#   source pbs/inc_conda_lengau.sh
#
# Optional overrides (exported before qsub or via qsub -v VAR=value):
#   LAPAI_CONDA_TRACK_A=/path/to/env   # default below (CHPC shared Anemoi)
#   LAPAI_CONDA_TRACK_B=name_or_path   # default lapai-credit (create via environment-credit-lengau.yml)

if [[ ! -f pbs/inc_conda_lengau.sh ]]; then
  echo "ERROR: cwd must be lapai-forecast repo root (needs ./pbs/). Now: $(pwd)" >&2
  exit 1
fi

module purge
module load chpc/python/anaconda/3-2024.10.1

CONDA_SH="/home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda.sh not found at $CONDA_SH — check module avail chpc/python/anaconda" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"

# Verified on Lengau (login nodes): interpreter has torch (+cu124 wheels) and anemoi imports.
export LAPAI_CONDA_TRACK_A="${LAPAI_CONDA_TRACK_A:-/apps/chpc/chem/anaconda3-2021.11/envs/anemoi-training}"
export LAPAI_CONDA_TRACK_B="${LAPAI_CONDA_TRACK_B:-lapai-credit}"

lapai_activate_track_a() {
  conda deactivate 2>/dev/null || true
  conda activate "${LAPAI_CONDA_TRACK_A}" || {
    echo "ERROR: conda activate failed: ${LAPAI_CONDA_TRACK_A}" >&2
    exit 1
  }
}

lapai_activate_track_b() {
  conda deactivate 2>/dev/null || true
  conda activate "${LAPAI_CONDA_TRACK_B}" || {
    cat <<'EOF' >&2
ERROR: TRACK_B conda env activation failed.

Create lapai-credit once (repo root):

  module load chpc/python/anaconda/3-2024.10.1
  source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
  conda env create -f environment-credit-lengau.yml
  conda activate lapai-credit
  pip install -e . --no-deps

Override for a different env:

  qsub -v LAPAI_CONDA_TRACK_B=my-env pbs/student.pbs
EOF
    exit 1
  }
}

lapai_load_cuda_gpu() {
  module load chpc/cuda/12.0/12.0 2>/dev/null || module load chpc/cuda/11.8/11.8 2>/dev/null || {
    echo "WARNING: CUDA module not loaded — expect torch.cuda=false here." >&2
  }
}

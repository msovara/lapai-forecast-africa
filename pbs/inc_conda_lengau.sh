#!/usr/bin/env bash
# Shared conda / module bootstrap for LapAI-Forecast PBS jobs on CHPC Lengau.
#
# Each PBS script must first:
#   cd "${PBS_O_WORKDIR}" || exit 1   # submit from lapai-forecast repo root
#   source pbs/inc_conda_lengau.sh
#
# Optional overrides (exported before qsub or via qsub -v VAR=value):
#   LAPAI_CONDA_TRACK_A=name_or_path   # default lapai-isolated Track A env (environment-anemoi.yml)
#   LAPAI_CONDA_TRACK_B=name_or_path   # default lapai-credit (environment-credit-lengau.yml)

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

# Isolated conda env built from environment-anemoi.yml (recommended on Lengau).
# Site-wide fallback (chem stack): LAPAI_CONDA_TRACK_A=/apps/chpc/chem/anaconda3-2021.11/envs/anemoi-training
export LAPAI_PBS_PROJECT="${LAPAI_PBS_PROJECT:-RCHPC}"
export LAPAI_CONDA_TRACK_B="${LAPAI_CONDA_TRACK_B:-lapai-credit}"

lapai_activate_track_a() {
  conda deactivate 2>/dev/null || true
  conda activate "${LAPAI_CONDA_TRACK_A}" || {
    cat <<'EOF' >&2
ERROR: TRACK_A conda env activation failed (default: lapai-anemoi).

Create the isolated env from repo root (after module load chpc/python/anaconda + source conda.sh):

  conda env create -f environment-anemoi.yml
  conda activate lapai-anemoi
  pip install -e . --no-deps

Or submit with the CHPC shared Anemoi stack (not isolated):

  qsub -v LAPAI_CONDA_TRACK_A=/apps/chpc/chem/anaconda3-2021.11/envs/anemoi-training pbs/trackA.pbs
EOF
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
  # GraphTransformer uses Triton kernels that JIT-compile with $CC. Lengau's
  # /bin/gcc is 4.8.5 and lacks <stdatomic.h>; load a modern GCC and point CC/CXX.
  module load gcc/9.2.0 2>/dev/null || module load gcc/7.3.0 2>/dev/null || {
    echo "WARNING: modern gcc module not loaded — Triton GT JIT may fail (stdatomic.h)." >&2
  }
  if command -v gcc >/dev/null 2>&1; then
    export CC="$(command -v gcc)"
    export CXX="$(command -v g++)"
    echo "[lapai] Triton build CC=${CC} ($(${CC} --version | head -1))"
  fi
}

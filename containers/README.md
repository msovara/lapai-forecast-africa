# Mvula containers (Apptainer / Singularity / Docker)

Reproducibility image for frozen Track B **v5** (`trackb-v5-c4e`).  
This is **Path A** (research / HPC / paper repro). Laptop demos without a container are **Path B** (`environment-mvula-enduser.yml` + `python run_mvula.py`) — see the repo root [README.md](../README.md) (“Which path should I use?”).

**Scope (same banner as `run_mvula.py info`):** validated use is analysis-forced African t2m (+6 h primary). **Not** supported: autonomous free-run, 10-day forecast, or full 65-channel state reconstruction. Checkpoint is **not** baked into the `.sif`.

## What is in the image

| Included | Not included |
|----------|----------------|
| Repo code, `reports/` artefacts, Streamlit + CPU PyTorch stack | Weights baked into `.sif` (bind-mount host `models/` from git clone) |
| `run_mvula.py` entry (`info` / `bench` / `dashboard`) | ARCO / ERA5 IC caches |
| Case A claim boundary in `%help` | Free-run / 10-day forecast |

Frozen v5 student is tracked in git as `models/student_global_stable_v5.ckpt` (~9 MiB). Bind-mount that directory into the container at run time (weights stay out of the image for reuse/flexibility).

## Files

| File | Role |
|------|------|
| [`Apptainer.def`](Apptainer.def) | **Canonical** Apptainer definition |
| [`Singularity.def`](Singularity.def) | Alias for CHPC docs that still say Singularity |
| [`Dockerfile`](Dockerfile) | Same CPU stack for Docker Desktop / CI |

Build **from the repository root** (so `%files` / `COPY` paths resolve).

## Build

### Apptainer / Singularity (Cassava, Lengau, Linux laptop)

```bash
cd /path/to/lapai-forecast-africa
git checkout trackb-v5-c4e

# Apptainer (preferred name)
apptainer build mvula-v5.sif containers/Apptainer.def

# Singularity CLI (same recipe)
singularity build mvula-v5.sif containers/Apptainer.def
```

Fakeroot / remote builds depend on site policy; on CHPC follow local Apptainer module docs if `build` needs elevated privileges.

### Docker

```bash
docker build -f containers/Dockerfile -t mvula-v5:cpu .
```

## Run

Place the v5 checkpoint under `./models/` on the host, then bind-mount:

```bash
# info (exit 1 if ckpt missing)
apptainer run -B "$PWD/models:/opt/lapai-forecast/models" mvula-v5.sif info

# CPU laptop-style bench → writes reports/MVULA_LAPTOP_BENCHMARK.json inside the
# container FS unless you also bind reports/; bind both to persist:
apptainer run \
  -B "$PWD/models:/opt/lapai-forecast/models" \
  -B "$PWD/reports:/opt/lapai-forecast/reports" \
  mvula-v5.sif bench

# Streamlit (publish port)
apptainer run --nv=false \
  -B "$PWD/models:/opt/lapai-forecast/models" \
  --env "STREAMLIT_SERVER_PORT=8501" \
  mvula-v5.sif dashboard -- --server.port 8501 --server.address 0.0.0.0
# then browse http://localhost:8501 (or SSH -L tunnel on HPC)
```

Docker equivalent:

```bash
docker run --rm \
  -v "$PWD/models:/opt/lapai-forecast/models" \
  -v "$PWD/reports:/opt/lapai-forecast/reports" \
  -p 8501:8501 \
  mvula-v5:cpu info
```

## Claim boundary

Frozen v5 is **analysis-forced African t2m** (Case A). It is **not** a free-running 10-day student forecast. See [`reports/FINAL_REPORT.md`](../reports/FINAL_REPORT.md) and [`reports/TRACKB_STATE_CLOSURE.md`](../reports/TRACKB_STATE_CLOSURE.md).

## Advanced (not default)

Re-running the full AF t2m expanded eval needs ARCO/network (and usually GPU). That is **not** the container `%runscript` default. Use host scripts such as `scripts/run_trackB_t2m_expanded_cassava.sh` on Cassava, or inspect packaged results under `reports/`.

## Path B (no container)

```bash
conda env create -f environment-mvula-enduser.yml
conda activate mvula-enduser
pip install -e ".[dev,data,ort]"
pip install -r requirements_streamlit.txt
python run_mvula.py info
python run_mvula.py bench
python run_mvula.py dashboard
# Windows: run_mvula.bat dashboard
```

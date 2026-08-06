# Track A full train — operator checklist

## Status — A1 PASSED (2026-08-06)

| Item | Result |
|------|--------|
| Gate | **`reports/TRACKA_A1_GATE.json` — passed 30/30** |
| Criterion | ≤5% RMSE degradation vs Phase 0 on t2m, u10, v10 @ +24h/+48h |
| Scorecard | `reports/TRACKA_COARSEN_SCORECARD.json` (extend-15k forecasts) |
| Checkpoint | `models/teacher_coarsened.ckpt` → extend run `…/b828d680-…/inference-last.ckpt` |
| Train path | 2000-step full (`trackA_coarsen_full`) then **+15000** (`trackA_coarsen_full_extend`) |
| Jobs | train **7357693** (gpu2005); forecast **7357781** / **7357782** (both Exit 0) |

t2m +24h is **better** than Phase 0 on all five Jan 2023 inits (~25–33% lower RMSE).  
`tp` is scored on the scorecard for reporting; it is **not** part of the A1 gate (enters at A2).

---

Zarr ready locally: `data/processed/lapai/era5_n96_2020_2021.zarr` (~26 GB, 2020–2021).

**Verified 2026-08-02** — build completed 2026-07-24 and the store is sound:
2924 timesteps, 2020-01-01T00 → 2021-12-31T18 (731 days × 4 at 6-hourly, so no gaps),
101 variables on the 40320-point O96 grid, no degenerate stdevs, no non-finite stats.
Worst fp16 headroom is `q_100` at 211 vs the 65504 limit. Re-check any rebuild with:

```bash
python scripts/diagnose_zarr_stats.py data/processed/lapai/era5_n96_2020_2021.zarr
```

## 1. Sync to Lengau

Large data goes on **lustre**, symlinked into the repo, so `~` stays small and the
PBS script still finds the Zarr at its expected relative path.

```bash
# On Lengau: destination + symlink (the -d guard in trackA_full.pbs follows symlinks)
mkdir -p /home/msovara/lustre/lapai-data
ln -s /home/msovara/lustre/lapai-data/era5_n96_2020_2021.zarr \
      ~/repos/lapai-forecast/data/processed/lapai/era5_n96_2020_2021.zarr
```

Windows has no `rsync`; use WSL (`rsync` 3.2.7). Two gotchas cost real time — see below.

```bash
# From WSL. ~3.4 MB/s observed => ~2h15m for 26 GB. Re-run to resume.
tr -d '\r' < scripts/upload_zarr_to_lengau.sh > /tmp/up.sh && bash /tmp/up.sh
```

**Gotcha 1 — DNS.** `lengau.chpc.ac.za` resolves to a private VPN address (10.128.15.127).
WSL is set to `generateResolvConf = false` with public nameservers, so the name is NXDOMAIN
inside WSL even though routing to the host works. Connect by IP and pin
`-o HostKeyAlias=lengau.chpc.ac.za` to keep one stable `known_hosts` entry.

**Gotcha 2 — backgrounding.** `nohup ... &` inside `wsl -e bash -lc` dies with rsync
exit 255 once the launching shell exits. Run rsync in the foreground of a persistent
WSL process instead.

Shell scripts piped to Lengau need `tr -d '\r'` first; the repo's CRLF endings break `bash -s`.

Confirm teacher warm-start exists on Lengau:
`~/repos/lapai-forecast/models/teacher_n320_gt6/inference.ckpt`

## 2. Submit train (Lengau login node)

```bash
cd ~/repos/lapai-forecast
# Optional dry-run first:
qsub -v LAPAI_TRACKA_DRY=1 pbs/trackA_full.pbs
# Full A1 (24h walltime, 2000 steps):
qsub pbs/trackA_full.pbs
qstat -u $USER
```

Checkpoint target: `models/teacher_coarsened.ckpt`  
Run logs under: `models/trackA_coarsen_full_runs/`

## 3. After train — forecast, score, gate

```bash
cd ~/repos/lapai-forecast
# Use lustre python if that is your train/infer stack
export LAPAI_PYTHON=/home/msovara/lustre/dev/lapai-anemoi/bin/python
$LAPAI_PYTHON scripts/run_trackA_closure.py
$LAPAI_PYTHON scripts/run_trackA_gate.py \
  --candidate reports/TRACKA_COARSEN_SCORECARD.json
```

Gate writes `reports/TRACKA_A1_GATE.json` (≤5% RMSE degradation vs Phase 0 at +24h/+48h for t2m, u10, v10).

## Config knobs

| Setting | Value |
|---------|--------|
| Config | `configs/trackA_coarsen_full.yaml` |
| Dataset | `era5_n96_2020_2021.zarr` |
| Train window | 2020-01-01 → 2021-09-12 |
| Val window | 2021-09-13 → 2021-12-31 |
| `max_steps` | 2000 (raise if job finishes early with headroom) |
| Walltime | 12:00:00 — `gpu_1` hard cap |
| Select | `1:host=gpu2005:ncpus=9:ngpus=1` (32 GiB V100; alt `gpu2006`) |

### Gate push (done) — extend after 2000 steps

2000 steps finished in ~8.5 min but **t2m +24h failed** A1 (cold bias). Extend fixed it:

```bash
bash scripts/submit_trackA_extend_32gb.sh   # +15000 steps from teacher_coarsened.ckpt
# retarget symlink → extend inference-last.ckpt
bash scripts/submit_trackA_forecast_extend.sh
# laptop: PYTHONPATH=. python scripts/rescore_trackA_gate_fast.py
```

Config: `configs/trackA_coarsen_full_extend.yaml`. Constant mean t2m debias alone was **not** enough — training was required.

`gpu_1` limits: `walltime <= 12:00:00`, `ncpus <= 10`, `ngpus = 1`, max 4 running /
10 queued per user. A 24h request is rejected at submit time with "Job violates queue
and/or server resource limits". If 2000 steps do not fit in 12h, resume from the last
checkpoint in `models/trackA_coarsen_full_runs/` rather than asking for a longer job.

**GPU memory (CHPC wiki):** most Lengau V100s are **16 GiB**. Only **gpu2005** and
**gpu2006** have **32 GiB**. Request them with an exact host pin, e.g.
`#PBS -l select=1:host=gpu2005:ncpus=9:ngpus=1` — not `node_type=32GB` (that fails with
`Insufficient amount of resource: nodetype` after the Aug 2026 reboot). Expect longer
queue wait for a specific host.

Verified dry-run (unpinned 16 GiB): job **7353454**, host `gpu4002`, Exit_status=0.

Smoke path unchanged: `qsub pbs/trackA.pbs` + `configs/trackA_coarsen.yaml`.

## Appendix — other data on Lengau lustre

`/home/msovara/lustre/era5-o96-1979-2023-6h-v8.zarr` looks tempting: same 101 variables
and same 40320-point O96 grid as our store, but 65744 timesteps (1979–2023) instead of 2924.

**It is not usable as-is.** Audited 2026-08-02 with `scripts/inspect_lustre_o96_zarr.sh`:

- 41822 of 65745 expected chunk files present — roughly 36% missing
- no `dates`, `latitudes`, `longitudes`, `mean` or `stdev` arrays
- no root `.zgroup` / `.zattrs` / `.zmetadata`
- one leftover `.partial` chunk and one zero-length chunk
- untouched since Nov 2025

It is an abandoned build. Finishing it means re-pulling ~24k timesteps from CDS. Worth
reviving only if we later want a long pre-training run — the grid and variable set already
match, so it would drop straight into `trackA_coarsen_full.yaml` if completed.

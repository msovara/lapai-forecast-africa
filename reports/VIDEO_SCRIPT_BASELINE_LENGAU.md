# Video script — LapAI baseline on Lengau (teacher inference)

**Purpose:** Walkthrough of [`RUNBOOK_BASELINE_LENGAU.md`](RUNBOOK_BASELINE_LENGAU.md) for team onboarding or async training.  
**Suggested length:** 18–22 minutes (full) · 8 minutes (minimal path only)  
**Recording setup:** Terminal font ≥ 14pt, dark theme, hide secrets (`~/.cdsapirc`, HF tokens).  
**Live presentation:** Open [`video_baseline_lengau.html`](video_baseline_lengau.html) in a browser and share screen in [Google Meet](https://meet.google.com/gox-emnj-hro).

---

## Before you record

| Item | Action |
|------|--------|
| Repo | `cd lapai-forecast && git pull` |
| Cluster access | SSH to Lengau login node |
| Env | `lapai-anemoi` already created (or show §1 setup once) |
| Checkpoint | Either pre-downloaded or demo download live |
| PBS project | Replace `CHPC` with your `-P` code in examples |

**Chapter markers (YouTube / Teams):**

| Time | Chapter |
|------|---------|
| 0:00 | Intro & goal |
| 1:30 | Environment (§1) |
| 4:00 | Download teacher checkpoint (§2–3) |
| 6:30 | Inference gate (§4 checklist steps 4–5) |
| 10:00 | GPU PBS job (§6.a) |
| 13:00 | Optional NetCDF output |
| 15:00 | ERA5 truth + scorecard (§8–8a) |
| 18:00 | Troubleshooting (§7) |
| 20:00 | Wrap-up |

---

## Scene 1 — Intro (0:00–1:30)

**On screen:** Title slide or repo README + runbook TOC.

**Narration:**

> This walkthrough covers the **baseline teacher inference path** on CHPC Lengau for the LapAI forecast project. By the end you will know how to activate the conda environment, download the AIFS teacher checkpoint, pass the local inference gate, submit a GPU PBS job, and optionally score a forecast against ERA5 over Africa.
>
> Everything lives in the **`lapai-forecast`** repo. The canonical checklist is **`reports/RUNBOOK_BASELINE_LENGAU.md`**.

**On screen:** Show minimal copy-paste block from runbook (lines 9–15).

---

## Scene 2 — Environment (1:30–4:00) · §1

**On screen:** Terminal on Lengau (login or `chpclic1`).

**Narration:**

> First we load Anaconda and activate **`lapai-anemoi`**. This env holds PyTorch, Anemoi training and inference, and the LapAI package.

**Type (slowly, pause after each block):**

```bash
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
conda activate lapai-anemoi
which python
python -c "import torch; print('torch', torch.__version__)"
```

**Narration (if env missing):**

> First-time setup should run on interactive node **`chpclic1`** — login nodes often OOM during conda solves. Request an interactive job, then run **`bash scripts/setup_lengau_envs.sh`**. Batch alternative: **`qsub pbs/setup_env_chpclic1.pbs`**.

**Show (optional, no live solve):**

```bash
qsub -I -P CHPC -q normal -l select=1:ncpus=4:mem=48GB -l walltime=8:00:00 \
  -l place=scatter:excl -W x=FLAGS:ADVRES:chpclic1
bash scripts/setup_lengau_envs.sh
```

---

## Scene 3 — Download checkpoint (4:00–6:30) · §2–3

**On screen:** `lapai-forecast/` root; open `configs/teacher_aifs.yaml` briefly (revision + filename).

**Narration:**

> The default teacher is the public AIFS checkpoint pinned in **`configs/teacher_aifs.yaml`**. Download weights into **`models/teacher`**.

```bash
cd ~/repos/lapai-forecast   # your clone path
git pull
python scripts/download_teacher_ckpt.py --local-dir models/teacher
ls -lh models/teacher/
```

**Narration:**

> Smoke-test that the file exists and PyTorch can read shallow keys:

```bash
python scripts/smoke_teacher_ckpt.py
```

**Expected on screen:** `OK` or key summary on stderr; no `MISSING`.

**Optional branch (15 sec):**

> For the gated challenge teacher **`C4E-Mvula/n320_gt6`**, run **`huggingface-cli login`** first, then use **`--config configs/teacher_n320_gt6.yaml`** on every download and inference step.

---

## Scene 4 — Inference gate (6:30–10:00) · checklist steps 4–5

**On screen:** Run the bundled gate script.

**Narration:**

> Before touching the GPU queue, we run **`lapai_inference_gate.sh`**. It chains three checks: smoke checkpoint, strict stack verify, and inference dry-run.

```bash
bash scripts/lapai_inference_gate.sh
```

**Pause — show output lines:**

1. `smoke_teacher_ckpt.py` — file present  
2. `verify_inference_stack.py --strict` — anemoi-inference on PATH  
3. `run_aifs_inference.py --dry-run` — YAML renders, command prints  

**Narration:**

> If this passes, you see **`OK: gate passed`**. On failure the script points you to **section 7** of the runbook — we'll cover that at the end.

**Show equivalent manual steps (quick montage):**

```bash
python scripts/verify_inference_stack.py --strict
python scripts/run_aifs_inference.py --dry-run
```

---

## Scene 5 — GPU PBS inference (10:00–13:00) · §6.a

**On screen:** `pbs/inference_aifs_teacher.pbs` header comments; then terminal.

**Narration:**

> Real inference needs a GPU. Submit from **`lapai-forecast`** repo root so **`PBS_O_WORKDIR`** is correct. Start with a PBS dry-run — it only renders YAML on the batch node:

```bash
qsub -v LAPAI_AIFS_INFER_DRY=1 pbs/inference_aifs_teacher.pbs
qstat -u $USER
```

**Narration:**

> Check the job log for **`--dry-run`** output and no errors. Then submit the full run:

```bash
qsub pbs/inference_aifs_teacher.pbs
```

**On screen:** Tail log file (`*.o*` / `*.e*` in PBS default output dir).

**Narration:**

> Default config uses **`configs/inference_aifs_minimal.yaml`** with test dataset and printer output — good for a first baseline. Log the run in the runs table at the bottom of section 6 in the runbook.

---

## Scene 6 — NetCDF artefact (13:00–15:00) · §6 optional

**On screen:** `configs/inference_aifs_netcdf_example.yaml` — highlight `output:` block.

**Narration:**

> For a saved forecast file, edit the NetCDF example YAML, uncomment **`output:`**, set path under **`data/processed/lapai/`**, then re-submit with a template variable:

```bash
qsub -v LAPAI_INFER_TEMPLATE=configs/inference_aifs_netcdf_example.yaml pbs/inference_aifs_teacher.pbs
```

**On screen:** Show expected output path after job completes (or pre-recorded file listing).

---

## Scene 7 — ERA5 truth & scorecard (15:00–18:00) · §8–8a

**On screen:** Split: CDS registration (browser) + terminal.

**Narration:**

> To score forecasts we need ERA5 ground truth for 2023–2025 over the Africa domain. One-time: register at Copernicus CDS and create **`~/.cdsapirc`**. Install extras with **`pip install -e '.[cds]'`**.

```bash
python scripts/download_era5_eval_truth.py --dry-run
python scripts/download_era5_eval_truth.py --years 2023
python scripts/download_era5_eval_truth.py --years 2023,2024,2025 --convert
```

**Narration:**

> Conversion needs cfgrib and eccodes — use a full env or workstation if Lengau nogrib env lacks GRIB.

```bash
python -m evaluation.run_scorecard \
  --pred data/processed/lapai/forecast.zarr \
  --truth data/processed/lapai/era5_eval_truth_africa.zarr \
  --variables t2m,tp,u10,v10 --temporal 6h,daily \
  --out reports/scorecard_africa.json --markdown
```

**On screen:** Open generated `reports/scorecard_africa.md` or JSON snippet.

---

## Scene 8 — Troubleshooting (18:00–20:00) · §7

**On screen:** Runbook §7 table.

**Narration (rapid FAQ):**

| Symptom | Fix (say aloud) |
|---------|-----------------|
| `teacher_ckpt_exists: MISSING` | Re-run download; path must match yaml |
| `anemoi-inference_path: MISSING` | `conda activate lapai-anemoi`; refresh env |
| `torch: NOT IMPORTABLE` | Don't use system Python on login node |
| Dry-run OK, real run fails | Use GPU PBS script; read stderr LAPAI lines |
| YAML rejected | Compare to Anemoi inference quickstart docs |

---

## Scene 9 — Wrap-up (20:00–21:00)

**On screen:** Minimal path block + links.

**Narration:**

> Recap: **`git pull`**, **`conda activate lapai-anemoi`**, download checkpoint, **`bash scripts/lapai_inference_gate.sh`**, then **`qsub pbs/inference_aifs_teacher.pbs`**. Optional: NetCDF template, ERA5 truth, scorecard. Full details stay in **`RUNBOOK_BASELINE_LENGAU.md`** and **`docs/LENGAU.md`**.

**End card:** Repo URL · runbook path · questions channel.

---

## Short cut — 8-minute version

Record only Scenes 1 (30s), 2 (1m), 3 (1.5m), 4 (2m), 5 (2.5m), 9 (30s). Skip ERA5/scorecard unless audience asks.

---

## Post-production checklist

- [ ] Blur/mute any API keys or tokens in terminal history  
- [ ] Add chapter markers from table above  
- [ ] Upload to shared drive; paste link in Meet chat or runbook  
- [ ] Append run date + video URL to runbook runs log (optional)

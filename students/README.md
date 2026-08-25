# LapAI student pathway

**Role:** Compressed O96 / laptop-deployable student distilled from the Track A teacher.

## Track A (closed for round-1 prune — accept K1)

- Report: `reports/TRACKA_REPORT.md` — **accept K1**, tp trade-off documented, **no A2 round 2**
- Teacher for distillation: `models/teacher_pruned.ckpt` (K1 from A1b+)
- Historical A1 notes: `reports/TEAM_NOTE_A1_PASSED.md`, `reports/TRACKA_A2_START.md`

## Track B (current)

- Start checklist: `reports/TRACKB_START.md`
- Configs: `configs/student_global.yaml`, `configs/student_distill.yaml` (`teacher_ckpt` → K1)
- Driver: `python training/train_student.py --config configs/student_global.yaml`
- Cassava smoke: `bash scripts/run_trackB_smoke_cassava.sh` (GPU 1)
- PBS (Lengau): `pbs/student.pbs`

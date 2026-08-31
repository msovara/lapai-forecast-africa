# Cassava node backup (laptop) — 2026-08-31

## Already present locally (verified SHA256 match vs Cassava)
- models/student_global_stable_v5.ckpt
  sha256: 5cf7080054bff60e88c1c9c74517cf5c471d34343f205967d3526171af5b17b3
- models/teacher_pruned.ckpt  (= K1 inference-last.ckpt)
  sha256: 5d1f805dbd546275c07b1a216c4ca500312ab9340cefb4f8071c50af20795ebe

## Copied here (small logs only)
- reports/cassava_node_backup/v5_run_logs/trackB_*v5*

## NOT copied (rebuild on new node if needed; too large for laptop)
- teacher_k1_cache_full2020_2021.zarr  (~79 GiB on Cassava)
- data/cache/student_ic_arco/           (~4.4 GiB on Cassava)
- Full Track A epoch intermediate ckpts under trackA_prune_k1_from_a1bplus_runs/
- Conda envs under /local/Mthetho/envs/ and miniforge3/

## Safe to wipe Cassava /local after this for close-out use of frozen v5
Rebuild train cache/envs on the new Slurm node only if you will retrain.

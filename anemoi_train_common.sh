#!/bin/bash
# Shared Anemoi training overrides for JRA-3Q on Lengau (sourced by PBS scripts).
# Reduces first-step cost (batch 8, 2 workers) and saves step checkpoints early.

anemoi_run_train() {
    local -a extra=("$@")
    echo "=== anemoi_run_train start: $(date) ==="
    free -h | head -3
    python -m anemoi.training train \
        data=zarr \
        data.frequency="24h" \
        data.timestep="24h" \
        data.forcing=[] \
        data.diagnostic=[] \
        data.normalizer.std=[] \
        data.normalizer.min-max=[] \
        data.normalizer.max=[] \
        data.normalizer.none=["hgt_1000","hgt_500","hgt_850","pres","rh_1000","rh_500","rh_850","tmp2m","tp","sdor","slor","cp","z","cos_latitude","cos_longitude","sin_latitude","sin_longitude","cos_julian_day","sin_julian_day","cos_local_time","sin_local_time","insolation","lsm"] \
        system.input.dataset="${COMBINED_DATASET}" \
        system.output.root="${MODEL_OUTPUT}" \
        dataloader.batch_size.training=8 \
        dataloader.batch_size.validation=1 \
        dataloader.limit_batches.validation=1 \
        dataloader.num_workers.training=2 \
        dataloader.num_workers.validation=2 \
        training.num_sanity_val_steps=0 \
        ~training.scalers.pressure_level \
        training.training_loss.scalers=["general_variable","node_weights"] \
        +diagnostics.plot.callbacks=[] \
        diagnostics.log.wandb.entity="anemoi-training" \
        diagnostics.log.mlflow.tracking_uri="file://${MODEL_OUTPUT}/mlruns" \
        diagnostics.checkpoint.every_n_train_steps.save_frequency=10 \
        diagnostics.checkpoint.every_n_train_steps.num_models_saved=3 \
    diagnostics.checkpoint.every_n_minutes.save_frequency=15 \
    +graph.edges.1.edge_builders.0.scale_resolutions=[1,2] \
    model=gnn \
        "${extra[@]}"
    local rc=$?
    echo "=== anemoi_run_train end: $(date) exit=${rc} ==="
    return $rc
}

anemoi_probe_zarr() {
    echo "=== Zarr read probe: $(date) ==="
    python "${WORK_DIR}/scripts/probe_zarr_read.py" "${COMBINED_DATASET}" || true
    echo "=== Zarr probe done: $(date) ==="
}

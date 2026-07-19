#!/usr/bin/env python3
"""Print Anemoi config keys related to memory / compile (run in anemoi-training env)."""
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

GlobalHydra.instance().clear()
initialize(config_path="pkg://anemoi.training/config", version_base=None)
cfg = compose(config_name="config", overrides=["model=gnn"])

paths = [
    "training.compile",
    "training.precision",
    "training.accum_grad_batches",
    "model.compile",
    "model.training",
    "hardware",
]
for p in paths:
    val = OmegaConf.select(cfg, p, default="<missing>")
    if val != "<missing>":
        print(f"=== {p} ===")
        print(OmegaConf.to_yaml(val)[:2000])

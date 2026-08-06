#!/usr/bin/env python3
"""Quick check that fixed tp selection returns finite Africa fields."""
from __future__ import annotations

import numpy as np

from evaluation.gcs_era5_truth import open_era5_at_valid_time

for target in [
    np.datetime64("2023-01-02T00:00:00"),
    np.datetime64("2023-01-02T06:00:00"),
    np.datetime64("2023-01-03T00:00:00"),
]:
    truth = open_era5_at_valid_time("tp", target)
    v = np.asarray(truth.values)
    fin = float(np.isfinite(v).mean())
    print(
        f"{target}: shape={v.shape} finite={fin:.4f}",
        end="",
    )
    if fin > 0:
        print(f" min={float(np.nanmin(v)):.6g} max={float(np.nanmax(v)):.6g}")
    else:
        print(" ALL NAN")

#!/usr/bin/env python3
"""Local-only t2m +24h bias: Track A vs Phase 0 forecasts (no GCS)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
INITS = ["20230101", "20230108", "20230115", "20230122", "20230129"]
LEAD_H = 24


def _step_for_lead(ds: xr.Dataset, lead_h: int) -> int:
    times = ds.time.values.astype("datetime64[ns]")
    # assume first time is +6h or +0h — match Phase0 helper if present
    dt_h = (times - times[0]) / np.timedelta64(1, "h")
    # lead relative to init: init = times[0] - first_step
    # Our NetCDFs usually start at init+6h. Prefer exact valid time = init+lead.
    # Infer init from filename elsewhere; here pick index where elapsed from first
    # is closest to lead - first_lead.
    first_lead = float(dt_h[0]) if len(dt_h) else 0.0
    # If first sample is already +6h, index for +24h is (24-6)/6 = 3
    target_from_first = lead_h - first_lead
    return int(np.argmin(np.abs(dt_h - target_from_first)))


def main() -> None:
    print(f"lead=+{LEAD_H}h  comparing Track A vs Phase 0 t2m\n")
    print(f"{'init':8s} {'bias_K':>8s} {'rmse_vs_p0':>10s} {'|bias|>2K%':>10s} {'cand_mean':>10s} {'p0_mean':>10s}")
    for init in INITS:
        ca = ROOT / f"data/processed/trackA/forecasts/{init}_00Z.nc"
        p0 = ROOT / f"data/processed/phase0/forecasts/{init}_00Z.nc"
        if not ca.exists():
            print(f"{init}: missing Track A forecast")
            continue
        if not p0.exists():
            print(f"{init}: missing Phase 0 forecast — skip spatial bias")
            continue
        dca = xr.open_dataset(ca)
        dp0 = xr.open_dataset(p0)
        i = _step_for_lead(dca, LEAD_H)
        # align times
        t_ca = dca.time.values[i]
        # nearest time in phase0
        j = int(np.argmin(np.abs(dp0.time.values.astype("datetime64[ns]") - t_ca)))
        a = np.asarray(dca["t2m"].isel(time=i).values, dtype=np.float64)
        b = np.asarray(dp0["t2m"].isel(time=j).values, dtype=np.float64)
        # regrid-nearest if shapes differ
        if a.shape != b.shape:
            print(f"{init}: shape mismatch TrackA={a.shape} Phase0={b.shape}")
            dca.close()
            dp0.close()
            continue
        mask = np.isfinite(a) & np.isfinite(b)
        diff = a - b
        bias = float(np.nanmean(diff[mask]))
        rmse = float(np.sqrt(np.nanmean(diff[mask] ** 2)))
        frac = float(np.mean(np.abs(diff[mask]) > 2.0))
        print(
            f"{init:8s} {bias:8.3f} {rmse:10.3f} {100*frac:9.1f}% "
            f"{float(np.nanmean(a[mask])):10.2f} {float(np.nanmean(b[mask])):10.2f}  "
            f"times ca={t_ca} p0={dp0.time.values[j]}"
        )
        dca.close()
        dp0.close()


if __name__ == "__main__":
    main()

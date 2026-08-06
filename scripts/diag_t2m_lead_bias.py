#!/usr/bin/env python3
"""t2m bias evolution: Track A vs Phase 0 across leads (local NetCDFs)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
INITS = ["20230101", "20230108", "20230115", "20230122", "20230129"]
LEADS_H = [6, 12, 24, 48, 72, 120]


def nearest_time_idx(ds: xr.Dataset, when: np.datetime64) -> int:
    times = ds.time.values.astype("datetime64[ns]")
    return int(np.argmin(np.abs(times.astype("int64") - np.datetime64(when, "ns").astype("int64"))))


def main() -> None:
    print("mean bias TrackA - Phase0 (K) by lead")
    print(f"{'lead':>6} " + " ".join(f"{i:>10}" for i in INITS) + f" {'mean':>10}")
    for lead in LEADS_H:
        row = []
        for init in INITS:
            ca = ROOT / f"data/processed/trackA/forecasts/{init}_00Z.nc"
            p0 = ROOT / f"data/processed/phase0/forecasts/{init}_00Z.nc"
            if not ca.exists() or not p0.exists():
                row.append(float("nan"))
                continue
            dca = xr.open_dataset(ca)
            dp0 = xr.open_dataset(p0)
            t0 = np.datetime64(f"{init[:4]}-{init[4:6]}-{init[6:8]}T00:00:00")
            when = t0 + np.timedelta64(lead, "h")
            # forecast files may start at +6h
            ia = nearest_time_idx(dca, when)
            ib = nearest_time_idx(dp0, when)
            a = np.asarray(dca["t2m"].isel(time=ia).values, dtype=np.float64)
            b = np.asarray(dp0["t2m"].isel(time=ib).values, dtype=np.float64)
            m = np.isfinite(a) & np.isfinite(b)
            row.append(float(np.mean(a[m] - b[m])) if m.any() else float("nan"))
            dca.close()
            dp0.close()
        mean = float(np.nanmean(row))
        print(f"+{lead:3d}h " + " ".join(f"{v:10.3f}" for v in row) + f" {mean:10.3f}")


if __name__ == "__main__":
    main()

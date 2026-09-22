# Contributing

Thanks for your interest in **Mvula** / LapAI-Forecast.

## What this repo is

A **research demonstrator** from ECMWF Code for Earth 2026 (African Stream): a compressed AIFS-lineage student for African short-range **t2m**, packaged for laptop CPU use.

It is **not** an operational NWP service, not a 10-day free-running forecast system, and not a smartphone app.

## Claim boundary (please respect in PRs / issues)

- Evaluation is **analysis-forced one-step** (native +6 h). Headline claim = **00Z** analysis IC.
- Packaged multi-hour columns are **IC-hour sensitivity** (00/06/12/18Z), not autoregressive lead skill.
- Case A (**Cout=3**): no full-state free-run / AR rollout without architecture change.
- **tp** skill is out of scope for the frozen v5 head.

See [`README.md`](README.md) and [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md).

## How to contribute

1. Open a [GitHub Issue](https://github.com/msovara/lapai-forecast-africa/issues) for bugs, questions, or proposals.
2. Prefer small PRs against `main` with a clear description of *why*.
3. Do not commit secrets (`.env`, CDS keys, tokens), large regenerable data (`.zarr`, GRIB), or private meeting notes.
4. Keep public messaging aligned with the claim boundary above.

## Licence

Code: Apache-2.0 · Docs/figures under `reports/` and `docs/`: CC BY 4.0 · Third-party teacher weights remain under their original terms ([`NOTICE`](NOTICE)).

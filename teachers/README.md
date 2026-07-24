# LapAI pathways (one project, two teachers)

```text
lapai-forecast/
  config/domains.yaml     # Africa lat [-40,40] lon [-20,70]  ← shared contract
  teachers/
    aifs/                 # Pathway A — Anemoi n320_gt6 Phase 0
    graphcast/            # Pathway B — GraphCast Africa (GCS + upstream repo)
  students/               # Track A O96 student (gates vs both teachers)
  evaluation/             # ONE scorecard API
```

| Pathway | Stack | Use |
|---------|-------|-----|
| **A — AIFS** | Anemoi / Lengau | Compression teacher + Track A baseline |
| **B — GraphCast** | GraphCast Africa | NaN-safe Africa baseline; 1° small model |
| **Student** | Track A O96 | Laptop-deployable target |

Integrate GraphCast **here** (adapter + shared domain), do not fork a second LapAI product.

# ViT-B E2-C seed-only bootstrap sensitivity analysis

Supplementary verification materials for Pattern Recognition manuscript PR-D-26-04876 (R2).

The analysis uses 24 archived E2-C ViT-B Layer-4 directional AUROC contrasts (four tested PE implementations, six seeds each). The supplied [input CSV](E2C_VITB_24_SEED_LEVEL_DELTAS.csv) matches the archived E2-C seed-level `DELTA_L4` values (24/24 exact comparisons).

**Method:** 200,000 seed-only percentile-bootstrap draws, NumPy PCG64 seed `20261111`, evaluation images fixed.

**Reproduce:** run `python -m pip install numpy` then `python recompute_vitb_seedonly_family_intervals.py` in this directory.

**Outputs:** [CSV](VITB_FAMILY_SEEDONLY_200K.csv) · [JSON](VITB_FAMILY_SEEDONLY_200K.json) · [Script](recompute_vitb_seedonly_family_intervals.py)

| PE implementation | Seed-only 95% CI |
|---|---|
| Learned | [-0.30844, -0.24893] |
| Sinusoidal | [-0.27385, -0.25337] |
| RoPE | [-0.33244, -0.21505] |
| ALiBi | [-0.28332, -0.11914] |

All four intervals remain negative. This is a sensitivity analysis, not a new training experiment. The original seed-plus-image intervals in manuscript Table 2 remain unchanged; 200,000 resamples do not create additional independent training seeds.

The same five materials are provided with the R2 submission in `PR_R2_VITB_SEED_ONLY_SENSITIVITY_2026-11-11.zip`.
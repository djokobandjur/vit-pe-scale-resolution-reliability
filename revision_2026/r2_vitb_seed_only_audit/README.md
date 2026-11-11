# Additional R2 ViT-B seed-only sensitivity check (2026-11-11)

This is a **post-release R2 statistical audit addendum** for Pattern Recognition manuscript PR-D-26-04876. It adds four seed-only percentile bootstrap confidence intervals for the ViT-B E2-C PE implementations. It does not modify Table 2's original hierarchical seed-plus-image confidence intervals, the six independent training seeds, any scientific estimate, or the failed original E6 replication gate.

**Inputs:** 24 paired directional Attention AUROC differences in [E2C_VITB_24_SEED_LEVEL_DELTAS.csv](E2C_VITB_24_SEED_LEVEL_DELTAS.csv), covering four PE implementations × six training seeds.  
**Method:** 200,000 seed-only percentile bootstrap replicates, NumPy PCG64 seed **20261111**, with evaluation images fixed.  
**Outputs:** [VITB_FAMILY_SEEDONLY_200K.csv](VITB_FAMILY_SEEDONLY_200K.csv) and [VITB_FAMILY_SEEDONLY_200K.json](VITB_FAMILY_SEEDONLY_200K.json).  
**Recompute:** `python -m pip install numpy`, then `python recompute_vitb_seedonly_family_intervals.py` in this directory.

| PE | 95% seed-only interval for E2-C directional AUROC contrast |
|---|---|
| Learned | [-0.30844, -0.24893] |
| Sinusoidal | [-0.27385, -0.25337] |
| RoPE | [-0.33244, -0.21505] |
| ALiBi | [-0.28332, -0.11914] |

## Provenance and publication status

The 24 seed-level contrast values in the supplied CSV were extracted from an **internal v0.3.2 working archive** during the R2 audit. Unlike a cross-archive proof, this does **not** claim to have independently confirmed byte-identical member paths or CSV checksums in the public v0.3.4 science-lock ZIP. The bundled 24 values are sufficient to independently rerun this bootstrap check. The CSV is **not** a model retraining or checkpoint inference replication.

This new check was produced on **2026-11-11**, **after** the immutable Zenodo 2.0.2 publication on 2026-11-10. It is **not contained in Zenodo 2.0.2** (https://doi.org/10.5281/zenodo.24117884). To ensure the journal receives the actual executed script and machine-readable outputs, the small five-file ZIP `PR_R2_VITB_SEED_ONLY_SENSITIVITY_AUDIT_2026-11-11.zip` should also be uploaded as an *additional verification/supplementary file in Editorial Manager*. The local ZIP has SHA-256 `ca79caeb1bfe7d3035270bed866d5449a7057b33fde45785adbcd8392b5aefea`.

**No new GitHub Release tag or Zenodo version is created for this limited post-release documentation update.**

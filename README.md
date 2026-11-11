# Resolution Sensitivity of Attention-Entropy OOD Detection in Vision Transformers: A Controlled Bidirectional Analysis

Reproducibility resources for **Pattern Recognition** manuscript **PR-D-26-04876**.

- **Stable Zenodo concept DOI (all versions):** https://doi.org/10.5281/zenodo.19386421
- **Latest published Zenodo version:** **2.0.2** (2026-11-10): https://doi.org/10.5281/zenodo.24117884
- **Previous R1 science-lock version:** **2.0.1** (2026-08-25): https://doi.org/10.5281/zenodo.22101504

## R2 E1 score-level reproducibility addendum (Zenodo 2.0.2)

The Zenodo **2.0.2** record contains **both** unchanged R1 science-lock evidence and the new R2 E1 attention-score archive:

| Archive | Contents | SHA-256 |
| --- | --- | --- |
| `PR_MAJOR_REVISION_REPLICATION_v0.3.4.zip` | Previous R1 science-lock replication package, retained unchanged | `d8d2d5bb5cae5c2d6d178f29dd435e3a91a5429aa4c0df3a7a7f99be0c3a6143` |
| `PR_R2_E1_ATTENTION_SCORE_ADDENDUM_v1.0.0.zip` | 48 E1 per-image attention NPZ files and 48 corresponding JSON summaries, SHA-256 manifest, and standalone numerical verification/audit scripts and outputs | `718c307e6e11113c6e0651b084d57673cbe55e85d1927f0fce788a00fcd20b8f` |

The E1 addendum independently recomputes **576/576** archived layer-level AUROC values from stored per-image attention scores (maximum absolute difference `2.22e-16`); pooled Layer-4 directional AUROC contrast is **+0.4142881933**. Coverage includes both native ImageNet-100↔CIFAR-100 directions, four tested positional-encoding implementations, six seeds, and 12 ViT-B layers.

**Scope:** This is a score-to-AUROC arithmetic verification, **not** an independent replication of training or checkpoint-to-score inference. No checkpoint, primary estimate, locked statistical criterion, or training/evaluation protocol is changed. The original **E6 random-PE prespecified replication gate remains FAIL**. The fixed-PE cross-scale intervals remain post-audit exploratory evidence.

See [R2 E1 addendum documentation](revision_2026/E1_R2_ADDENDUM.md) and [Zenodo 2.0.2 release notes](revision_2026/RELEASE_NOTES_v2.0.2.md) for provenance details. The 27.4 MB addendum ZIP is deposited on **Zenodo**, not duplicated as a GitHub repository binary.

## R1 science-lock context

The previous science-lock archive (`v0.3.4`) includes all eight architecture×PE Layer-4 F→C-vs-C→F matched-seed Pearson correlations. The two largest magnitudes have opposite signs (Sinusoidal ViT-B `r=+0.945`; RoPE ViT-S `r=-0.947`), so this is treated as a descriptive shared-seed/nuisance diagnostic, not PE-specific mechanistic evidence.

The version **2.0.2** update preserves those interpretations and the originally failed E6 gate; it only improves availability and auditability of E1 score-level evidence.

## Additional R2 ViT-B seed-only sensitivity audit (2026-11-11)

The post-Zenodo-2.0.2 ViT-B E2-C family bootstrap check (24 seed-level contrasts, NumPy seed 20261111, 200,000 bootstrap draws, original Table 2 CIs unchanged) is available as [a separately versioned, standalone GitHub audit](revision_2026/r2_vitb_seed_only_audit/README.md). **This new audit was not part of immutable Zenodo 2.0.2 (published 2026-11-10)**. For the R2 journal submission, the corresponding small verification ZIP is also provided separately for Editorial Manager upload.

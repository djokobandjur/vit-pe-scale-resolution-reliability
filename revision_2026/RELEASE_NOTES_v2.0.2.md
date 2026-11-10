# R2 E1 score-level addendum — Zenodo 2.0.2 (published 2026-11-10)

**Zenodo version DOI:** https://doi.org/10.5281/zenodo.24117884  
**All-version concept DOI:** https://doi.org/10.5281/zenodo.19386421

## Added

- 48 native E1 per-image attention-entropy NPZ files and 48 JSON summaries, covering two transfer directions, four tested PE implementations, six seeds, and 12 layers.
- SHA-256 manifests and verification scripts. Rank-based recomputation matches **576/576** E1 layer-level AUROC values to floating-point precision (maximum absolute difference **2.22e-16**); pooled Layer-4 directional contrast **+0.4142881933**.
- R2 score-level and bootstrap-resampling sensitivity audits.

## Preserved

- Existing `PR_MAJOR_REVISION_REPLICATION_v0.3.4.zip` science-lock archive, unchanged.
- All previously locked experimental protocols, checkpoints, estimates and inference status.
- The original E6 random-PE prespecified replication gate **FAIL**.
- The status of fixed-PE cross-scale inference as exploratory/post-audit.

## Files and checksums

- `PR_MAJOR_REVISION_REPLICATION_v0.3.4.zip`: SHA-256 `d8d2d5bb5cae5c2d6d178f29dd435e3a91a5429aa4c0df3a7a7f99be0c3a6143`
- `PR_R2_E1_ATTENTION_SCORE_ADDENDUM_v1.0.0.zip`: SHA-256 `718c307e6e11113c6e0651b084d57673cbe55e85d1927f0fce788a00fcd20b8f`

The files are available through the **Zenodo record** rather than duplicated in GitHub. This document records the already-published Zenodo version; no separate GitHub Release tag is created, to avoid unintended automated Zenodo duplication.

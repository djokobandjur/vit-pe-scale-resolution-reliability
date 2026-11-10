# R2 E1 attention-score evidence addendum — published 2026-11-10

**Manuscript:** PR-D-26-04876 (Pattern Recognition), R2 minor revision.

**Published Zenodo release:** **2.0.2** — https://doi.org/10.5281/zenodo.24117884  
**Stable all-version concept DOI:** https://doi.org/10.5281/zenodo.19386421  
**Prior R1 science-lock:** **2.0.1** — https://doi.org/10.5281/zenodo.22101504

## Deposited archives

- **Prior science-lock, preserved unchanged:** `PR_MAJOR_REVISION_REPLICATION_v0.3.4.zip`; SHA-256 `d8d2d5bb5cae5c2d6d178f29dd435e3a91a5429aa4c0df3a7a7f99be0c3a6143`
- **New E1 addendum:** `PR_R2_E1_ATTENTION_SCORE_ADDENDUM_v1.0.0.zip`; SHA-256 `718c307e6e11113c6e0651b084d57673cbe55e85d1927f0fce788a00fcd20b8f`

Both archives are attached to the **published Zenodo 2.0.2 record**; no large binary ZIP is committed to this Git repository.

## Contents and scope

The new addendum includes the `E1 attention NPZ.zip` nested archive (48 per-image attention-entropy NPZ files and 48 run-level `SUMMARY.json` records), SHA-256 manifest, read-only rank-based AUROC verification script, a machine-readable JSON result, and R2 audits for E1 and controlled experiments.

E1 coverage: ImageNet-100→CIFAR-100 and CIFAR-100→ImageNet-100 native transfer, four tested PE implementations, six seeds, and all 12 ViT-B layers. Full input validation cohorts comprise 5,000 ImageNet-100 and 10,000 CIFAR-100 images.

The verifier matches **576/576** archived E1 layer-level AUROCs (maximum absolute difference **2.22e-16**). Pooled Layer-4 directional contrast remains **+0.4142881933**.

**Scope boundary:** This is a computation audit starting from stored per-image scores. It does **not** independently repeat model training or checkpoint-to-score inference. No R1/R2 primary estimate, checkpoint, prespecified gate, or locked protocol changed. The original **E6 random-PE replication gate remains FAIL**. Harmonized fixed-PE cross-scale inference remains an exploratory post-audit estimand.

## Reproduction

Download `PR_R2_E1_ATTENTION_SCORE_ADDENDUM_v1.0.0.zip` from [Zenodo 2.0.2](https://doi.org/10.5281/zenodo.24117884), verify its SHA-256, and read the included `README.md` and `verify_e1_score_to_auroc.py` before running the verification. The verification inputs and outputs are included in that archive.

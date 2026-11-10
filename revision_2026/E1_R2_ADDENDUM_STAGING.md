# R2 E1 attention-score evidence addendum (publication staging)

**Manuscript:** PR-D-26-04876 (Pattern Recognition), R2 minor revision

**Status:** Staged for a new Zenodo record version; **not yet published**. The current public science-lock full replication archive remains `PR_MAJOR_REVISION_REPLICATION_v0.3.4.zip` (SHA-256 `d8d2d5bb5cae5c2d6d178f29dd435e3a91a5429aa4c0df3a7a7f99be0c3a6143`).

**New additive archive:** `PR_R2_E1_ATTENTION_SCORE_ADDENDUM_v1.0.0.zip`  
**SHA-256:** `718c307e6e11113c6e0651b084d57673cbe55e85d1927f0fce788a00fcd20b8f`  
**Existing Zenodo Concept DOI:** https://doi.org/10.5281/zenodo.19386421  
**New version DOI:** pending Zenodo publication.

## Contents and scope

The addendum includes an unmodified copy of `E1 attention NPZ.zip` (48 per-image attention-entropy NPZ files and 48 per-run `SUMMARY.json` records), a member-level SHA-256 manifest, read-only rank-based AUROC verification script, a reproducible JSON result, and R2 audit outputs for E1 and the controlled experiments.

E1 coverage: two native ImageNet-100/CIFAR-100 transfer directions, four tested positional-encoding implementations, six training seeds, and all 12 Transformer layers. Input arrays use the complete 5,000 ImageNet-100 validation images and 10,000 CIFAR-100 test images in the relevant roles.

The score-to-AUROC verifier matches **576/576** archived layer-level AUROC numbers (maximum absolute difference **2.22e-16**). The pooled Layer-4 directional gap remains **+0.4142881933**. Checkpoint IDs and input manifest hashes are supplied and were reconciled against the v0.3.2 native provenance evidence in the internal audit.

**Limits:** This verifies arithmetic from stored per-image scores. It is not independent checkpoint-to-score inference and is not model retraining. No locked E1/E2-C/E6 result, checkpoint, or prespecified gate changes. The original E6 random-PE gate remains **FAIL**, and fixed-PE cross-scale inference remains exploratory/post-audit.

## Publication sequence

1. Create a **new version** under the existing Zenodo concept record (do not overwrite the published v0.3.4 archival evidence).
2. Import/preserve the previous science-lock archive, and add the `PR_R2_E1_ATTENTION_SCORE_ADDENDUM_v1.0.0.zip` file.
3. Check deposited archive SHA-256 `718c307e6e11113c6e0651b084d57673cbe55e85d1927f0fce788a00fcd20b8f`; publish after metadata review.
4. Record the resulting *version-specific* Zenodo DOI in this file and in the top-level README. Keep the stable concept DOI and v0.3.4 checksum as historical pointers.
5. Consider whether a separate GitHub release tag would trigger a duplicate Zenodo deposition through the GitHub integration before creating a release.

This staging document must not be represented as a completed Zenodo publication before the DOI and archive are verified.

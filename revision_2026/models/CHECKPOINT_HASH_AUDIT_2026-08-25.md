# Checkpoint SHA-256 audit — 2026-08-25

## Result

**PASS for manifest structure and hash coverage.**

- Central revision checkpoint rows: **120/120**
- Previously locked native ViT-B hashes: **48**
- Newly recorded hashes from local USB export: **72**
- Remaining empty SHA-256 fields: **0**
- Unique SHA-256 values: **120/120**
- Duplicate checkpoint keys in the 72-row USB manifest: **0**
- Duplicate SHA-256 values in the 72-row USB manifest: **0**
- Invalid SHA-256 strings: **0**
- Non-positive file sizes: **0**

## 72-row USB cohort composition

- ViT-B / CIFAR-100 / coarse: **24**
- ViT-S / CIFAR-100 / native: **24**
- ViT-S / CIFAR-100 / coarse: **24**

Total: **72**

The two ViT-S smoke-test checkpoints were intentionally excluded before hash generation by restricting the source to `VITS_CIFAR_2X2\TRAINING`.

## Provenance boundary

The 72 SHA-256 values were computed locally from the user's FMLE-exported checkpoint files and supplied in:

`PR_R1_72CHECKPOINTS_SHA256_USB_2026-08-25.csv`

SHA-256 of that CSV:

`c1ea072764b66fa3374799aca3807a67983b90a5963f27d98bab335271cf61f1`

Total byte size represented by its 72 checkpoint rows:

`12292348572` bytes

This audit validates the manifest structure, cohort mapping, hash syntax, uniqueness, and completeness. It does **not** independently re-hash the checkpoint binaries, because the 72 `.pth` files themselves were not uploaded into this environment.

Public checkpoint URLs remain pending until the external model archive is uploaded.

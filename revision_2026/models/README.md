# Revision model checkpoints

Expected central checkpoint count: **120**.

```text
models/
├── ViTB/
│   ├── ImageNet100/native/   # 4 PE × 6 seeds = 24
│   └── CIFAR100/
│       ├── native/           # 24
│       └── coarse/           # 24
└── ViTS/
    └── CIFAR100/
        ├── native/           # 24
        └── coarse/           # 24
```

The authoritative index is `MODEL_CHECKPOINT_MANIFEST.csv`.

Current public status:
- **120/120 checkpoint SHA-256 hashes are populated.**
- 48 native ViT-B hashes come from the locked audit provenance.
- 72 ViT-B coarse and ViT-S native/coarse hashes were computed locally from the FMLE-exported checkpoint files on 2026-08-25 and recorded in `PR_R1_72CHECKPOINTS_SHA256_USB_2026-08-25.csv`.
- **120/120 public checkpoint locations are populated and Drive-side hash verification is complete.**
- `public_url` points to the stable dataset/cohort folder that contains each checkpoint.

### E6 cross-validation

The 48 ViT-S E6 checkpoint hashes are additionally cross-validated against the original E6 48/48 run ledger, the E6 SHA-256 manifest, and the E6 validation record. All 48 paths and hashes agree. See `E6_CHECKPOINT_PROVENANCE_CROSSCHECK_2026-08-25.md`.

### Verified public checkpoint locations (2026-08-25)

- ViT-B / ImageNet-100 / native (24): `https://drive.google.com/drive/folders/1QH3EG9mf6oSWwzwrhdp599S5VB_hShZx`
- ViT-B / CIFAR-100 / native (24): `https://drive.google.com/drive/folders/1pPgYxOtHfBRwvuSPKCwSzbynNN65tSv7`
- ViT-B / CIFAR-100 / coarse plus ViT-S / CIFAR-100 / native+coarse (72): `https://drive.google.com/drive/folders/1amyzkeeN2MxWiDm9mR7dA2llxsFCgS2m`

Drive-side verification completed before public release: 48/48 native ViT-B checkpoints matched the locked provenance hashes, and 72/72 revision-specific checkpoints matched the verified USB source copies.

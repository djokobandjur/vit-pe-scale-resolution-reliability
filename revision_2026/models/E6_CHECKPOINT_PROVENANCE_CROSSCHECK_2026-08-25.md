# E6 checkpoint provenance cross-check — 2026-08-25

## Result

**PASS — 48/48 E6 final training checkpoints cross-validated across three provenance surfaces.**

- E6 validation JSON: PASS
- expected / observed / passed runs: 48 / 48 / 48
- failed runs: 0
- E6 run-ledger rows: 48
- E6 SHA-256 manifest rows: 48
- authoritative 120-checkpoint manifest E6 rows: 48
- path/key matches: 48/48
- SHA-256 matches, authoritative manifest ↔ E6 run ledger: 48/48
- SHA-256 matches, authoritative manifest ↔ E6 SHA-256 manifest: 48/48
- unique E6 SHA-256 values: 48/48

Protocol ID: `PR-E6-VITS-CIFAR-2X2-v1.0`

Model source SHA-256:

`83fc337128dec7f896c9816842806789a634154dea8372bb0a43bae19188d3bf`

The two ViT-S smoke-test checkpoints are outside the 48-run training ledger and are not part of the authoritative revision checkpoint manifest.

## Preserved provenance inputs

- `provenance/E6_48CHECKPOINTS_SHA256.txt`
- `provenance/E6_48OF48_RUN_LEDGER.csv`
- `provenance/E6_48OF48_VALIDATION.json`

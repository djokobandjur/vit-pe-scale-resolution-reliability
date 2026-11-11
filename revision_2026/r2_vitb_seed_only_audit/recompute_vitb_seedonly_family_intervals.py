#!/usr/bin/env python3
"""Seed-only ViT-B E2-C family-level AUROC-delta bootstrap sensitivity.
Input: bundled 24 seed-level contrasts, originally extracted from internal
v0.3.2 working evidence; public v0.3.4 byte-equivalence is not asserted.
Does not replace original seed-plus-image intervals in Table 2.
"""
from pathlib import Path
import csv, json
import numpy as np

ROOT = Path(__file__).resolve().parent
PE = ('learned', 'sinusoidal', 'rope', 'alibi')
SEEDS = (42, 123, 456, 789, 1011, 1213)
N_BOOT = 200000
RNG_SEED = 20261111

with (ROOT / 'E2C_VITB_24_SEED_LEVEL_DELTAS.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
lut = {(r['pe_type'], int(r['seed'])): float(r['Attention_Delta']) for r in rows}
assert len(rows) == len(lut) == 24
A = np.array([[lut[(p, s)] for s in SEEDS] for p in PE], dtype=np.float64)
rg = np.random.default_rng(RNG_SEED)
chunks = []
for _ in range(N_BOOT // 10000):
    idx = rg.integers(0, 6, (10000, 4, 6))
    chunks.append(np.take_along_axis(A[None, :, :], idx, axis=2).mean(axis=2))
vals = np.concatenate(chunks)
results = []
for i, p in enumerate(PE):
    lo, hi = np.quantile(vals[:, i], [.025, .975])
    results.append({'pe': p, 'mean_delta': float(A[i].mean()),
                    'ci_lower': float(lo), 'ci_upper': float(hi)})
lo, hi = np.quantile(vals.mean(axis=1), [.025, .975])
results.append({'pe': 'pooled_equal_weight', 'mean_delta': float(A.mean()),
                'ci_lower': float(lo), 'ci_upper': float(hi)})

with (ROOT / 'VITB_FAMILY_SEEDONLY_200K.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['pe', 'mean_delta', 'ci_lower', 'ci_upper'])
    writer.writeheader()
    writer.writerows(results)
(ROOT / 'VITB_FAMILY_SEEDONLY_200K.json').write_text(json.dumps({
    'method': 'percentile bootstrap on six paired seed-level directional AUROC contrasts per PE; evaluation images fixed',
    'rng': 'numpy.random.default_rng (PCG64)', 'rng_seed': RNG_SEED,
    'draws': N_BOOT,
    'source': '24 included E2-C ViT-B seed-level directional contrasts (E2C_VITB_24_SEED_LEVEL_DELTAS.csv)',
    'source_provenance': 'Extracted from internal v0.3.2 working evidence; not independently cross-checked against the public v0.3.4 ZIP',
    'status': 'additional exploratory sensitivity; original Table 2 CIs not replaced',
    'results': results
}, indent=2) + '\n')
print(json.dumps(results, indent=2))

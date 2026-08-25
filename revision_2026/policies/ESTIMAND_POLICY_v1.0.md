# ESTIMAND_POLICY_v1.0

## Decision
For manuscript-facing pooled comparisons, Learned, Sinusoidal, RoPE, and ALiBi are a **fixed, prespecified design factor**. The pooled estimand is the equal-weight mean across these four families.

Inference resamples **seeds within each PE family** and then averages the four family means equally. PE families themselves are not resampled.

## Rationale
These four mechanisms were deliberately selected; they are not a random sample from a population of all possible PE mechanisms. A random-PE hierarchy targets generalization to unseen PE families, which is a different and much broader estimand.

## Cross-architecture comparison
ViT-S minus ViT-B is paired by identical PE family and seed, with the same resampled seed indices used in both architectures.

## Transparency
The original E6 prelocked replication gate used a random-PE hierarchy and remains **FAIL**. It is not overwritten. The harmonized fixed-PE analysis is a **post-audit estimand harmonization**, not a retroactive preregistration.

Thus:
- `ORIGINAL_PRELOCKED_RANDOM_PE_GATE = FAIL`
- `HARMONIZED_FIXED_PE_L4_ZERO_EXCLUSION = PASS`

## Scope limitation
Fixed-PE inference supports conclusions about these four prespecified mechanisms and their equal-weight average. It does not statistically generalize to unseen PE families.

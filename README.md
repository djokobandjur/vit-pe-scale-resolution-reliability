# Resolution Sensitivity of Attention-Entropy OOD Detection in Vision Transformers: A Controlled Bidirectional Analysis

> Pattern Recognition manuscript **PR-D-26-04876**, major-revision evidence snapshot: **2026-08-25**.

## Overview

This repository accompanies our revised study of **attention entropy as a training-free internal out-of-distribution (OOD) monitoring signal in Vision Transformers (ViTs)**.

The revision studies four prespecified positional-encoding (PE) families — **Learned, Sinusoidal, RoPE, and ALiBi** — with **six seeds per PE family**. Rather than asking which PE is globally “best,” the revised paper asks a narrower reliability question:

> **When does attention entropy provide a transferable OOD signal, and when does its score direction or magnitude depend on the shift construction, network depth, PE family, or model scale?**

The revised central evidence consists of four linked experiments:

| Experiment | Architecture | Design | Main purpose |
|---|---|---|---|
| **E1** | ViT-B | Native ImageNet-100 ↔ CIFAR-100 | Establish the native bidirectional OOD phenomenon |
| **E2-C** | ViT-B | Controlled CIFAR-100 native/coarse 2×2 | Test whether the native direction persists when semantics and final input geometry are controlled |
| **E3** | ViT-B | Same E1/E2-C samples and checkpoints | Compare attention entropy with MSP and Energy |
| **E6** | ViT-S | Controlled CIFAR-100 native/coarse 2×2 | Test transfer across conventional ViT model scale |

All central cohorts use seeds:

`{42, 123, 456, 789, 1011, 1213}`

## Main findings of the major revision

### 1. Native cross-domain attention entropy is strongly directional

At the prespecified Layer-4 endpoint in E1, the equal-weight mean across the four PE families gives a native directional AUROC gap of:

**Δ_native = +0.414**
95% CI **[+0.402, +0.426]**

This native comparison is intentionally treated as **confounded**: ImageNet-100 and CIFAR-100 differ in semantics, image statistics, native resolution, training distribution, and preprocessing.

### 2. The controlled CIFAR-100 experiment reverses the direction

E2-C uses the same CIFAR-100 semantic dataset and identical final 32×32 input / 8×8 patch-grid geometry in both directions. The controlled intervention is:

- **native:** original 32×32 CIFAR-100
- **coarse:** 32→16→32 using bilinear interpolation with antialiasing
- **F→C:** native-trained model evaluated on coarse inputs
- **C→F:** coarse-trained model evaluated on native inputs

At Layer 4 in ViT-B:

**Δ_asym = AUROC(F→C) − AUROC(C→F) = −0.252**
95% CI **[−0.279, −0.226]**

All **24/24 PE×seed cells are negative**.

This falsifies the prespecified positive-direction hypothesis and shows that the E1 native asymmetry is **not a universal resolution law**.

### 3. Attention entropy is not reducible to standard output confidence

On the exact same samples and checkpoints:

| Score | E1 native directional gap | E2-C controlled Δ |
|---|---:|---:|
| **Attention entropy** | +0.414 | −0.252 |
| **MSP** | +0.118 | +0.058 |
| **Energy** | +0.088 | −0.019 |

MSP does not mirror the controlled reversal. Energy weakly mirrors its sign, but at much smaller magnitude.

This is a diagnostic comparison, **not** a claim that attention entropy is universally superior to feature-space or hybrid OOD methods.

### 4. Depth and score orientation matter

The controlled effect is strongly layer-dependent. Layer 4 remains the primary endpoint and, after completion of the full layer analysis, is also the **only one of the 12 prespecified secondary layers with no AUROC<0.5 score-orientation inversions** across all tested PE×seed cells, both directions, and both ViT-B and ViT-S.

Layer 2 has a larger absolute directional contrast, but much of that magnitude is associated with score inversion for Learned, Sinusoidal, and ALiBi. **RoPE is the reproducible early-layer exception**, with 0/6 F→C inversions at Layer 2 in both architectures.

The specific Layer-2 inversion interpretation is secondary/post hoc and does not replace the Layer-4 primary endpoint.

### 5. ViT-S preserves the pooled reverse effect, but not uniformly across PE families

Using the harmonized fixed-PE estimand in E6:

**ViT-S Layer-4 pooled Δ = −0.127**
95% CI **[−0.175, −0.084]**

Learned, Sinusoidal, and ALiBi remain negative at the family level, whereas RoPE is heterogeneous.

The **original broader random-PE pre-specified replication gate remains FAIL**. The harmonized fixed-PE result is a narrower post-audit estimand and is not presented as a retroactive gate pass.

Across all 12 layers, the pooled ViT-B and ViT-S directional profiles remain strongly aligned:

- Pearson **r = 0.933**
- cosine similarity **= 0.940**

## Interpretation boundary

The revised paper supports the following scoped conclusion:

> **Attention-entropy OOD discriminability is structured but not universal: it depends on shift direction and construction, network depth, PE family, and architecture scale.**

The fixed-PE pooled analyses apply to the **four prespecified PE mechanisms studied here** and do not statistically generalize to unseen PE families.

The revised paper does **not** claim:

- a universal resolution-asymmetry law;
- a global ranking of PE families;
- universal superiority of RoPE or relative PE;
- that attention entropy replaces standard OOD detectors;
- safety-critical reliability without deployment-specific validation;
- architecture universality beyond the tested ViT-B and ViT-S settings.

## Revision-specific repository layout

The revision materials are organized under `revision_2026/`.

```text
revision_2026/
├── README.md
├── derived/
│   ├── MASTER_EXPERIMENT_SUMMARY.json
│   ├── MASTER_NUMBER_LEDGER.csv
│   ├── MASTER_CLAIM_LEDGER.csv
│   ├── MASTER_GATE_LEDGER.json
│   ├── HARMONIZED_FIXED_PE_L4_ANALYSIS.json
│   ├── L2_SECONDARY_ANALYSIS.json
│   ├── LAYER_SCORE_ORIENTATION_AUDIT.csv
│   ├── SEVERITY_PROXY_ANALYSIS.json
│   ├── BASELINE_INTERPRETATION_v2.json
│   └── CROSS_ARCHITECTURE_LAYER_PROFILE_COMPARISON_v2.json
├── policies/
│   ├── ESTIMAND_POLICY_v1.0.md
│   └── LAYER_STATUS_POLICY_v1.0.md
├── analysis/
│   └── recompute_harmonized_analysis_v1.py
├── models/
│   ├── README.md
│   └── MODEL_CHECKPOINT_MANIFEST.csv
├── runtimes/
│   └── RUNTIME_PACKAGES_MANIFEST.csv
├── replication/
│   └── README.md
└── checksums/
    └── REVISION_SHA256.txt
```

The complete audited evidence bundle is:

`PR_MAJOR_REVISION_REPLICATION_v0.3.2.zip`

SHA-256:

`74801b899972af7f4f9ea550dad7daf6564af2b05d6deb7358826eb1d75640a2`

Because this archive is about 92 MB and sits close to GitHub's per-file size limit, the recommended public location for the full immutable bundle is **Zenodo**, while the lightweight summary/derived files above remain directly browsable in GitHub.

## Model checkpoints

The central revision uses **120 distinct `best_model.pth` checkpoints**:

| Cohort | Count |
|---|---:|
| ViT-B, ImageNet-100, native training | 24 |
| ViT-B, CIFAR-100, native training | 24 |
| ViT-B, CIFAR-100, coarse training | 24 |
| ViT-S, CIFAR-100, native training | 24 |
| ViT-S, CIFAR-100, coarse training | 24 |
| **Total** | **120** |

Each cohort contains:

`4 PE families × 6 seeds = 24 checkpoints`

### Public checkpoint locations

- **ViT-B / ImageNet-100 / native (24):** https://drive.google.com/drive/folders/1QH3EG9mf6oSWwzwrhdp599S5VB_hShZx
- **ViT-B / CIFAR-100 / native (24):** https://drive.google.com/drive/folders/1pPgYxOtHfBRwvuSPKCwSzbynNN65tSv7
- **ViT-B / CIFAR-100 / coarse + ViT-S / CIFAR-100 / native+coarse (72):** https://drive.google.com/drive/folders/1amyzkeeN2MxWiDm9mR7dA2llxsFCgS2m

The file `revision_2026/models/MODEL_CHECKPOINT_MANIFEST.csv` is the authoritative index. It records architecture, dataset, training condition, PE family, seed, relative path, SHA-256 for all 120 central checkpoints, and public location.

All **120/120 central checkpoint SHA-256 hashes and public locations are populated and verified**. The 48 native ViT-B checkpoints were hash-verified against the existing public Drive cohorts; the 72 revision-specific ViT-B coarse / ViT-S native+coarse checkpoints were hash-verified from FMLE export → USB selection → public Drive copy.

## Full replication archive / Zenodo

### Stable Zenodo concept DOI

**https://doi.org/10.5281/zenodo.19386421**

This Concept DOI represents all archived releases and resolves to the latest Zenodo version. The GitHub `v2.0.0` release is the major-revision release associated with the revised manuscript.

Recommended immutable asset:

`PR_MAJOR_REVISION_REPLICATION_v0.3.2.zip`

The archive includes raw/derived E1, E2-C, E3, and E6 outputs; exact runtime packages; manifests; checksums; audit records; estimand/layer policies; and lineage information.

### Zenodo version history

Earlier Zenodo versions under the same Concept DOI preserve the original-submission provenance. Version-specific DOIs remain immutable; the Concept DOI above is the stable citation target for the evolving software record.

## Reproducing the revised evidence

There are three levels of reproduction.

### A. Verify the reported manuscript numbers

Use the lightweight files under `revision_2026/derived/` together with:

`revision_2026/analysis/recompute_harmonized_analysis_v1.py`

The current single source of truth for manuscript-facing numbers is the audited `PR_MAJOR_REVISION_REPLICATION_v0.3.2` package.

### B. Re-run the evaluation/analysis pipelines

Exact runtime packages are indexed in:

`revision_2026/runtimes/RUNTIME_PACKAGES_MANIFEST.csv`

The primary packages are:

- E1 native OOD evaluation
- E2-C controlled ViT-B training/evaluation
- E3 MSP/Energy baseline evaluation
- E6 ViT-S native/coarse training/evaluation

Each package is checksum-locked.

### C. Re-run training from scratch

Training from scratch requires CIFAR-100 and the ImageNet-100 subset construction described in the paper. ImageNet data are not redistributed by this repository.

The controlled CIFAR pipeline uses **MixUp (α=0.8)** and **does not use CutMix**.

## Legacy original-submission materials

The root-level scripts, `table1_scripts/`, `reproduce_paper_results.ipynb`, and `original_paper_results/` are retained for provenance of the initially submitted manuscript.

They are **not** the source of truth for the major revision.

In particular, the following original-submission interpretations have been superseded by the controlled revision experiments:

- universal “resolution asymmetry”;
- global superiority of relative PE;
- “Orthogonality Trap” as a validated central construct;
- “spatial builder / conservator” taxonomy as a main conclusion;
- n=3 central OOD statistics.

For the revised manuscript, use `revision_2026/` and the revision-specific model/archive links.

## Citation

The manuscript is currently under major revision at *Pattern Recognition*.

**Manuscript:** PR-D-26-04876
**Revised title:** *Resolution Sensitivity of Attention-Entropy OOD Detection in Vision Transformers: A Controlled Bidirectional Analysis*

For the software/replication record, cite the stable Zenodo Concept DOI: `10.5281/zenodo.19386421`. A journal-article citation should be added after publication.

## Authors

1. Djoko Bandjur
2. Milos Bandjur
3. Aleksandar Micic — ORCID: https://orcid.org/0000-0001-8606-3528
4. Branimir Jaksic

## Contact

Corresponding author: djoko.bandjur@pr.ac.rs

For questions about the released revision artifacts, please use the repository issue tracker or contact the corresponding author.

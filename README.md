# Scale, Resolution, and Reliability: A Comparative Information-Theoretic Analysis of Positional Encodings in Vision Transformers

[![DOI](https://zenodo.org/badge/1190775611.svg)](https://doi.org/10.5281/zenodo.19386421)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the code, links to trained model weights and experimental logs for our paper submitted to **Pattern Recognition** (Elsevier).

We present a comprehensive information-theoretic framework for comparing four positional encoding (PE) strategies in Vision Transformers: **Learned**, **Sinusoidal**, **RoPE**, and **ALiBi**. Unlike prior work that evaluates PE strategies solely through classification accuracy, we deploy layer-wise Mutual Information, Shannon entropy, a Spatial Decodability Protocol, and CKA representation similarity analysis across 24 independently trained ViT-Base models on ImageNet-100 and CIFAR-100.

### Key Findings

- **Layer 4** is a consistent zero-shot OOD detection checkpoint (AUROC up to 0.989)
- A **resolution asymmetry** bounds attention-based OOD detection: reliable only when the target domain is coarser than training data
- **Relative PE** (RoPE, ALiBi) provides superior structural reliability over absolute encodings
- The **Orthogonality Trap** — validated via CKA — identifies a failure mode where positional and semantic streams remain non-interacting (Sinusoidal PE)
- **Spatial builders** (RoPE, ALiBi) construct emergent spatial representations, while **spatial conservators** (Sinusoidal) preserve static positional codes

---

## Main Results

### Classification Accuracy (Top-1 %)

| PE Strategy | ImageNet-100 | CIFAR-100 |
|-------------|-------------|-----------|
| Learned | 79.4 ± 0.5 | 68.3 ± 0.3 |
| Sinusoidal | 81.5 ± 0.3 | 66.9 ± 0.4 |
| **RoPE** | **84.5 ± 0.3** | **73.3 ± 0.2** |
| ALiBi | 81.1 ± 0.3 | 67.7 ± 0.4 |

### OOD Detection (Direction A: IN-100 → C-100)

| PE Strategy | AUROC_mid | Best Layer (Peak AUROC) |
|-------------|-----------|------------------------|
| Learned | 0.841 ± 0.017 | L4 (0.949) |
| Sinusoidal | 0.879 ± 0.013 | L4 (0.971) |
| **RoPE** | **0.947 ± 0.002** | **L4 (0.989)** |
| ALiBi | 0.884 ± 0.010 | L4 (0.941) |

### CKA(L1, L12) — Orthogonality Trap Indicator

| PE Strategy | CKA(L1, L12) | Interpretation |
|-------------|-------------|----------------|
| **Sinusoidal** | **0.974 ± 0.002** | Least transformation (Orthogonality Trap) |
| ALiBi | 0.949 ± 0.005 | Moderate |
| Learned | 0.938 ± 0.009 | Active transformation |
| RoPE | 0.929 ± 0.018 | Most transformation (spatial builder) |

---

## 🛠️ Reproduction Steps (Google Colab)

To reproduce the results presented in the paper, we recommend using **Google Colab**.
> [!IMPORTANT]
> **Note on Local Execution**
> This repository is optimized for Google Colab. The scripts contain hardcoded absolute paths. To run this project locally, you must perform a search for these paths and update all directory-related variables to match your local environment.

---

### **Step 1 --- Google Drive Preparation**
*   Create a folder named `pe_experiment` in your root Google Drive directory.
*   **Final Path on Drive:** `/My Drive/pe_experiment/`
*   **Note:** In Colab, the full path will be: `/content/drive/MyDrive/pe_experiment/`

---

### **Step 2 --- Data Setup & Structure**

⚠️ **IMPORTANT:** The folder structure must be identical to the diagram below. All script paths are hardcoded.

1.  **From GitHub:** Download the repository and copy the following files into the root folder `/pe_experiment/`:
    *   The `table1_scripts` folder (including all 14 scripts).
    *   Python scripts: `00_setup_imagenet.py`, `full_scale_experiment.py` and `cka_colab_trap.py`.
    *   Text files: `imagenet100_classes.txt` and `val_labels.txt`.
    *   The Colab notebook: `reproduce_paper_results.ipynb` and `cka_colab_notebook.ipynb`.

2.  **ImageNet Dataset Acquisition:** Create a folder named `imagenet` within `/pe_experiment/`. 
    *   ImageNet requires registration. Go to image-net.org and register with an academic email. Once approved, you will receive an email with a unique download link. Use it to download the ILSVRC2012_img_val.tar archive (~6.3 GB).
    *   Place the archive inside /imagenet/. **Do NOT extract the archive.** The Colab script will handle the extraction automatically.

---

### 📥 Download Trained Model Weights
1. **[ImageNet-100 Models](https://drive.google.com/drive/folders/1gPwVSE0qctWVeaGwCv3eGQdQR4IK6Xds?usp=sharing)**
2. **[CIFAR-100 Models](https://drive.google.com/drive/folders/16pEAbdH4aRpw-3s2vm4TbMey1GPQn2FQ?usp=sharing)**

> [!IMPORTANT]
> **Instruction:** Select all -> **"Make a copy"** -> Move copies to `/pe_experiment/results/` (or `results_cifar100/`). Ensure that each model's individual   **subdirectory** is preserved and contains both `best_model.pth` and `training_history.json`.

    *   ⚠️ Note: The folder /pe_experiment/results/ must also contain the two shared analysis files: adversarial_pe_results.json and analysis_data.json (see the diagram).
    *   ⚠️ Note: The folder /pe_experiment/results_cifar100/ must also contain the shared file: adversarial_pe_results_cifar100.json (see the diagram).

---

### **Step 3 --- Launch in Colab**
*   Open the **`reproduce_paper_results.ipynb`** notebook from your Google Drive.
*   **Runtime:** Go to **Runtime > Change runtime type** and select **GPU (H100 or A100)**.
*   **Mount & Setup:** Run the **first cell (Cell 1)**. 
    *   When the popup appears, click **"Connect to Google Drive"** to grant access.
    *   **Note:** This cell will automatically mount your Drive and **copy all necessary scripts from your Drive into the Colab local environment (SSD)**.

---

### **Step 4 --- Code Execution**
*   **Starting from Cell 2:** Run the remaining cells one by one in the given order, from top to bottom.
*   **Sequential Execution:** Ensure each cell finishes completely before starting the next one to maintain the correct data flow.
*   **Automatic Saving:** The scripts will automatically create the **`/figures/`** and **`/table1/`** directories within **`/pe_experiment/results/`** and **save all generated outputs (figures and json files) directly into them.**

   **For CKA validation of the Orthogonality Trap** run the cka_colab_notebook.ipynb.

### **Verification:**
Once execution is complete, you can compare your generated outputs in /results/figures/ and /results/table1/ with the original study results provided in the original_paper_results/ folder for consistency.

---

#### **STRUCTURE BEFORE REPRODUCTION**
```text
🗂️ /My Drive/pe_experiment/
├──🗂️ table1_scripts/              # (14 scripts)
├──🗂️ imagenet/
│   └── ILSVRC2012_img_val.tar     # (Keep archived!)
├──🗂️ results/                     # ImageNet100 results
│   ├──🗂️ alibi_seed42/
│       ├── best_model.pth
│       └── training history.json 
│   ├──🗂️ alibi_seed123/
│   ├──🗂️ alibi_seed456/
│   ├──🗂️ learned_seed42/
│   ├──🗂️ learned_seed123/
│   ├──🗂️ learned_seed456/
│   ├──🗂️ rope_seed42/
│   ├──🗂️ rope_seed123/
│   ├──🗂️ rope_seed456/
│   ├──🗂️ sinusoidal42/
│   ├──🗂️ sinusoidal123/
│   ├──🗂️ sinusoidal456/
│   ├── analysis_data.json
│   └── adversarial_pe_results.json
│
│
├──🗂️ results_cifar100/           # CIFAR100 results
│   ├──🗂️ alibi_seed42/
│       ├── best_model.pth
│       └── training history.json 
│   ├──🗂️ alibi_seed123/
│   ├──🗂️ alibi_seed456/
│   ├──🗂️ learned_seed42/
│   ├──🗂️ learned_seed123/
│   ├──🗂️ learned_seed456/
│   ├──🗂️ rope_seed42/
│   ├──🗂️ rope_seed123/
│   ├──🗂️ rope_seed456/
│   ├──🗂️ sinusoidal42/
│   ├──🗂️ sinusoidal123/
│   ├──🗂️ sinusoidal456/
│   └── adversarial_pe_results_cifar100.json
│    
├── imagenet100_classes.txt
├── val_labels.txt
├── cka_orthogonality_trap.py
├── cka_colab_notebook.ipynb        
├── 00_setup_imagenet.py
├── full_scale_experiment.py
└── reproduce_paper_results.ipynb
```
---

## Model Architecture

All models use ViT-Base configuration:

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| MLP ratio | 4.0 |
| Dropout | 0.1 |
| Patch size (ImageNet-100) | 16×16 (196 patches) |
| Patch size (CIFAR-100) | 4×4 (64 patches) |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Learning rate | 3×10⁻⁴ |
| Weight decay | 0.1 |
| Schedule | Cosine with 20-epoch warmup |
| Batch size | 128 |
| Epochs | 300 |
| Label smoothing | 0.1 |
| MixUp α | 0.8 |
| CutMix α | 1.0 |
| Seeds | {42, 123, 456} |

---



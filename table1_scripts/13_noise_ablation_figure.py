"""
SCRIPT 13 — Figure 5: PE Noise Ablation Study
===============================================
Generate Figure 5 from analysis_data.json 

Input:
  results/analysis_data.json

Output:
  results/figures/figure5_noise_ablation.png  (300 DPI)
  results/figures/figure5_noise_ablation.pdf

Run in Colab:
    %run /content/13_noise_ablation_figure.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT   = '/content/drive/My Drive/pe_experiment'
DATA_PATH    = os.path.join(DRIVE_ROOT, 'results', 'analysis_data.json')
OUT_FIG_DIR  = os.path.join(DRIVE_ROOT, 'results', 'figures')
os.makedirs(OUT_FIG_DIR, exist_ok=True)

PE_TYPES     = ['learned', 'sinusoidal', 'rope', 'alibi']
PE_LABELS    = {'learned': 'Learned', 'sinusoidal': 'Sinusoidal',
                'rope': 'RoPE', 'alibi': 'ALiBi'}
PE_COLORS    = {'learned': '#E24B4A', 'sinusoidal': '#185FA5',
                'rope': '#1D9E75', 'alibi': '#BA7517'}
PE_MARKERS   = {'learned': 'o', 'sinusoidal': 's', 'rope': '^', 'alibi': 'D'}

SEEDS        = ['42', '123', '456']
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

# ── Load data ────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

print(f"Ucitano: {DATA_PATH}")

# ── Create figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# ── Panel 1: Apsolutna tacnost ────────────────────────────────────────────────
ax = axes[0]
for pe in PE_TYPES:
    all_accs = []
    for s in SEEDS:
        if s in data[pe]:
            accs = data[pe][s]['noise_ablation']['accuracies']
            all_accs.append(accs)
    if not all_accs:
        continue

    means = np.mean(all_accs, axis=0)
    stds  = np.std(all_accs,  axis=0)

    ax.plot(NOISE_LEVELS, means,
            color=PE_COLORS[pe], marker=PE_MARKERS[pe],
            markersize=5, linewidth=1.8, label=PE_LABELS[pe], zorder=3)
    ax.fill_between(NOISE_LEVELS, means - stds, means + stds,
                    alpha=0.15, color=PE_COLORS[pe], zorder=2)

ax.axvline(x=1.0, color='#888888', linestyle=':', linewidth=1.0, alpha=0.7)
ax.text(1.05, 82, 'Critical\nthreshold', fontsize=7.5, color='#888888', va='top')

ax.set_xlabel('Noise Scale (σ)', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_title('ImageNet-100: Accuracy under PE Noise', fontsize=11,
             pad=10, color='#333333')
ax.set_xscale('symlog', linthresh=0.1)
ax.set_xticks(NOISE_LEVELS)
ax.set_xticklabels([str(x) for x in NOISE_LEVELS], fontsize=8)
ax.set_ylim(0, 92)
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10, framealpha=0.9, loc='lower left',
          frameon=True, edgecolor='#dddddd')

# ── Panel 2: Relativna tacnost ────────────────────────────────────────────────
ax2 = axes[1]
for pe in PE_TYPES:
    all_rel = []
    for s in SEEDS:
        if s in data[pe]:
            accs      = np.array(data[pe][s]['noise_ablation']['accuracies'])
            clean_acc = accs[0]
            rel       = (accs / clean_acc) * 100
            all_rel.append(rel)
    if not all_rel:
        continue

    means = np.mean(all_rel, axis=0)
    stds  = np.std(all_rel,  axis=0)

    ax2.plot(NOISE_LEVELS, means,
             color=PE_COLORS[pe], marker=PE_MARKERS[pe],
             markersize=5, linewidth=1.8, label=PE_LABELS[pe], zorder=3)
    ax2.fill_between(NOISE_LEVELS, means - stds, means + stds,
                     alpha=0.15, color=PE_COLORS[pe], zorder=2)

ax2.axhline(y=50, color='#888888', linestyle=':', linewidth=1.0, alpha=0.7)
ax2.text(0.12, 51, '50% threshold', fontsize=7.5, color='#888888', va='bottom')
ax2.axvline(x=1.0, color='#888888', linestyle=':', linewidth=1.0, alpha=0.7)

ax2.set_xlabel('Noise Scale (σ)', fontsize=12)
ax2.set_ylabel('Relative Accuracy (% of clean)', fontsize=12)
ax2.set_title('ImageNet-100: Relative Robustness to PE Noise', fontsize=11,
              pad=10, color='#333333')
ax2.set_xscale('symlog', linthresh=0.1)
ax2.set_xticks(NOISE_LEVELS)
ax2.set_xticklabels([str(x) for x in NOISE_LEVELS], fontsize=8)
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle(
    'Figure 5: Noise Ablation Study — PE Robustness\n'
    'Gaussian noise added to positional encodings at test time '
    '(shaded = ±σ, 3 seeds)',
    fontsize=12, y=1.01, color='#1a1a1a'
)
plt.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
png_path = os.path.join(OUT_FIG_DIR, 'figure5_noise_ablation.png')
pdf_path = os.path.join(OUT_FIG_DIR, 'figure5_noise_ablation.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path,           bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"✓ PNG: {png_path}")
print(f"✓ PDF: {pdf_path}")

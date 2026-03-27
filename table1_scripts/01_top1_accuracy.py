"""
SCRIPT 01 — Top-1 Accuracy (μ ± σ over 3 seeds)
================================================
Reads training_history.json for all 24 models and extracts max(val_acc).

Run in Colab:
    %run /content/01_top1_accuracy.py

Output: results/table1/top1_accuracy.json
"""

import os, json
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT = '/content/drive/My Drive/pe_experiment'
RESULTS_IN = os.path.join(DRIVE_ROOT, 'results')           # ImageNet-100 checkpoints
RESULTS_CF = os.path.join(DRIVE_ROOT, 'results_cifar100')  # CIFAR-100 checkpoints
OUT_DIR    = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_DIR, exist_ok=True)

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS    = [42, 123, 456]

# ── Helper function ───────────────────────────────────────────────────────────
def load_best_val_acc(results_root: str, pe_type: str, seed: int):
    """Load training_history.json and return the maximum validation accuracy."""
    path = os.path.join(results_root, f'{pe_type}_seed{seed}', 'training_history.json')
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    with open(path) as f:
        hist = json.load(f)
    return float(np.max(hist['val_acc']))

# ── Main logic ────────────────────────────────────────────────────────────────
output = {}

for dataset_name, results_root in [('imagenet100', RESULTS_IN), ('cifar100', RESULTS_CF)]:
    print(f"\n{'='*55}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*55}")
    output[dataset_name] = {}

    print(f"\n{'PE Type':<14} {'Seed42':>9} {'Seed123':>9} {'Seed456':>9}  {'μ':>7} {'σ':>7}")
    print("-" * 62)

    for pe in PE_TYPES:
        vals, per_seed = [], {}
        for seed in SEEDS:
            acc = load_best_val_acc(results_root, pe, seed)
            per_seed[seed] = acc
            if acc is not None:
                vals.append(acc)

        if vals:
            mu, sigma = np.mean(vals), np.std(vals)
            output[dataset_name][pe] = {
                'per_seed': per_seed,
                'mean': round(mu, 4),
                'std':  round(sigma, 4),
            }
            seed_str = "  ".join(
                f"{per_seed[s]:6.2f}" if per_seed[s] is not None else "   N/A"
                for s in SEEDS
            )
            print(f"{pe:<14} {seed_str}  {mu:7.2f} {sigma:7.4f}")
        else:
            print(f"{pe:<14} {'N/A':>9} {'N/A':>9} {'N/A':>9}  {'N/A':>7} {'N/A':>7}")

# ── Save results ──────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'top1_accuracy.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved: {out_path}")

# ── Print in table format ─────────────────────────────────────────────────────
print("\n\n── TABLE 2 FORMAT ────────────────────────────────────────────────────────")
print(f"{'Metric':<28} {'Dataset':<10} {'Learned':>15} {'Sinusoidal':>15} {'RoPE':>15} {'ALiBi':>15}")
print("-" * 90)
for ds_label, ds_key in [('IN-100', 'imagenet100'), ('C-100', 'cifar100')]:
    row = f"{'Top-1 Acc (%)':<28} {ds_label:<10}"
    for pe in PE_TYPES:
        if pe in output.get(ds_key, {}):
            d   = output[ds_key][pe]
            row += f"  {d['mean']:5.2f}±{d['std']:.2f}  "
        else:
            row += f"  {'N/A':^13}  "
    print(row)

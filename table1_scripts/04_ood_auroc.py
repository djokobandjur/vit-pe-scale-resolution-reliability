"""
SCRIPT 04 — OOD AUROC from existing adversarial JSON files
===========================================================
Reads from pre-saved files:
  - results/adversarial_pe_results.json           (ImageNet-100)
  - results_cifar100/adversarial_pe_results_cifar100.json

Output: results/table1/ood_auroc.json

Run in Colab:
    %run /content/04_ood_auroc.py
"""

import os, json
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT  = '/content/drive/My Drive/pe_experiment'
ADV_IN_PATH = os.path.join(DRIVE_ROOT, 'results', 'adversarial_pe_results.json')
ADV_CF_PATH = os.path.join(DRIVE_ROOT, 'results_cifar100',
                            'adversarial_pe_results_cifar100.json')
OUT_DIR     = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_DIR, exist_ok=True)

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS    = [42, 123, 456]

# ── AUROC from degradation curve ──────────────────────────────────────────────
def auroc_from_degradation_curve(clean_acc: float, perturbed: dict) -> float:
    """
    Args:
      clean_acc  : accuracy without perturbation (%)
      perturbed  : {epsilon_str: acc} — FGSM-PE or PGD-PE curve

    Logic:
      - X axis: epsilon values normalised to [0, 1]
      - Y axis: accuracy / clean_acc (relative, in [0, 1])
      - AUROC = area under that curve (trapezoidal integration)

    A model that retains accuracy across all perturbation levels → AUROC near 1.
    A model that collapses immediately → AUROC near 0.
    """
    epsilons = sorted(float(e) for e in perturbed.keys())
    accs     = [perturbed[str(e)] for e in epsilons]
    eps_max  = max(epsilons)
    eps_norm = [e / eps_max for e in epsilons]
    acc_norm = [min(a / clean_acc, 1.0) for a in accs]  # clamp to 1.0

    # Prepend clean point (0, 1.0)
    xs = [0.0] + eps_norm
    ys = [1.0] + acc_norm
    return round(float(np.trapz(ys, xs)), 4)

def compute_combined_auroc(seed_data: dict) -> dict:
    """Compute AUROC separately for FGSM and PGD curves, plus variance_gini."""
    clean     = seed_data['clean_acc']
    fgsm_auroc = auroc_from_degradation_curve(clean, seed_data['fgsm_pe'])
    pgd_auroc  = auroc_from_degradation_curve(clean, seed_data['pgd_pe'])
    vta_auroc  = None
    if 'vta' in seed_data:
        vta_auroc = auroc_from_degradation_curve(clean, seed_data['vta'])
    return {
        'fgsm_auroc':    fgsm_auroc,
        'pgd_auroc':     pgd_auroc,
        'vta_auroc':     vta_auroc,
        'mean_auroc':    round(np.mean([fgsm_auroc, pgd_auroc]), 4),
        'variance_gini': seed_data.get('variance_gini'),
        'clean_acc':     clean,
    }

# ── Process one dataset ───────────────────────────────────────────────────────
def process_dataset(adv_path: str, ds_name: str) -> dict:
    if not os.path.exists(adv_path):
        print(f"  [MISSING] {adv_path}"); return {}

    with open(adv_path) as f:
        adv_data = json.load(f)

    result = {}
    print(f"\n{'='*65}")
    print(f"Dataset: {ds_name.upper()}  |  {adv_path}")
    print(f"{'='*65}")
    print(f"\n{'PE Type':<14} {'Seed42':>10} {'Seed123':>10} {'Seed456':>10}"
          f"  {'μ (FGSM)':>10} {'σ':>8}")
    print("-" * 70)

    for pe in PE_TYPES:
        if pe not in adv_data:
            print(f"{pe:<14}  N/A"); continue

        fgsm_aurocs, pgd_aurocs, ginis = [], [], []
        per_seed = {}

        for seed in SEEDS:
            seed_str = str(seed)
            if seed_str not in adv_data[pe]:
                per_seed[seed] = None; continue
            metrics = compute_combined_auroc(adv_data[pe][seed_str])
            per_seed[seed] = metrics
            fgsm_aurocs.append(metrics['fgsm_auroc'])
            pgd_aurocs.append(metrics['pgd_auroc'])
            if metrics['variance_gini'] is not None:
                ginis.append(metrics['variance_gini'])

        if fgsm_aurocs:
            mu_f, sig_f = np.mean(fgsm_aurocs), np.std(fgsm_aurocs)
            mu_p, sig_p = np.mean(pgd_aurocs),  np.std(pgd_aurocs)
            mu_g, sig_g = (np.mean(ginis), np.std(ginis)) if ginis else (None, None)

            result[pe] = {
                'per_seed':       per_seed,
                'fgsm_auroc':     {'mean': round(mu_f,4), 'std': round(sig_f,4)},
                'pgd_auroc':      {'mean': round(mu_p,4), 'std': round(sig_p,4)},
                'variance_gini':  {'mean': round(mu_g,4), 'std': round(sig_g,4)}
                                  if mu_g else None,
                'ood_auroc_mean': round(np.mean([mu_f, mu_p]), 4),
                'ood_auroc_std':  round(np.mean([sig_f, sig_p]), 4),
            }
            seeds_str = "  ".join(
                f"{per_seed[s]['fgsm_auroc']:8.4f}" if per_seed.get(s) else "     N/A"
                for s in SEEDS
            )
            print(f"{pe:<14} {seeds_str}  {mu_f:10.4f} {sig_f:8.4f}")
            if mu_g:
                print(f"  {'(gini)':12} {' '*30}  {mu_g:10.4f} {sig_g:8.4f}")

    return result

# ── Run ───────────────────────────────────────────────────────────────────────
output = {}
output['imagenet100'] = process_dataset(ADV_IN_PATH, 'imagenet100')
output['cifar100']    = process_dataset(ADV_CF_PATH, 'cifar100')

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'ood_auroc.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved: {out_path}")

# ── Print in table format ─────────────────────────────────────────────────────
print("\n\n── TABLE 2 FORMAT ────────────────────────────────────────────────────────")
print(f"{'Metric':<28} {'Dataset':<10} {'Learned':>15} {'Sinusoidal':>15}"
      f" {'RoPE':>15} {'ALiBi':>15}")
print("-" * 90)

for ds_label, ds_key in [('IN-100', 'imagenet100'), ('C-100', 'cifar100')]:
    for metric_label, metric_key in [
        ('Robustness FGSM', 'fgsm_auroc'),
        ('Robustness PGD',  'pgd_auroc'),
        ('Variance Gini',   'variance_gini'),
    ]:
        row = f"{metric_label:<28} {ds_label:<10}"
        for pe in PE_TYPES:
            d = output.get(ds_key, {}).get(pe)
            if d and metric_key in d and d[metric_key]:
                m    = d[metric_key]
                row += f"  {m['mean']:.4f}±{m['std']:.4f}  "
            else:
                row += f"  {'N/A':^13}  "
        print(row)

print("\n✓ Completed")

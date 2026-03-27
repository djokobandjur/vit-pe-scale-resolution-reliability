"""
SCRIPT 07 — Assemble Table 2 from all JSON files
=================================================
Reads all JSON files from results/table1/ and prints the complete Table 2.
Run AFTER all preceding scripts (01-06).

Output: results/table1/TABLE1_FINAL.txt  +  TABLE1_FINAL.csv

Run in Colab:
    %run /content/07_assemble_table1.py
"""

import os, json, csv
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT = '/content/drive/My Drive/pe_experiment'
TABLE_DIR  = os.path.join(DRIVE_ROOT, 'results', 'table1')

PE_TYPES  = ['learned', 'sinusoidal', 'rope', 'alibi']
PE_LABELS = {'learned': 'Learned', 'sinusoidal': 'Sinusoidal',
             'rope': 'RoPE', 'alibi': 'ALiBi'}
SEEDS     = [42, 123, 456]

# ── Load all JSON files ───────────────────────────────────────────────────────
def load_json(filename):
    path = os.path.join(TABLE_DIR, filename)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}"); return {}
    with open(path) as f:
        return json.load(f)

top1    = load_json('top1_accuracy.json')
entropy = load_json('attention_entropy.json')
mi      = load_json('mutual_information.json')
ood     = load_json('ood_auroc.json')
probe   = load_json('linear_probe.json')
ib      = load_json('ib_compression.json')

# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt(data, ds_key: str, pe: str, sub_key: str = None, decimals: int = 2) -> str:
    """Extract mean ± std from the standard JSON structure."""
    if not data: return 'N/A'
    d = data.get(ds_key, {}).get(pe)
    if d is None: return 'N/A'
    if sub_key:
        d = d.get(sub_key)
        if d is None: return 'N/A'
    mu  = d.get('mean')
    std = d.get('std')
    if mu is None: return 'N/A'
    f = f"{{:.{decimals}f}}"
    return f"{f.format(mu)} ± {f.format(std if std else 0)}"

def fmt_ood(ood_data, ds_key, pe, metric='fgsm_auroc', decimals=3) -> str:
    d = ood_data.get(ds_key, {}).get(pe)
    if d is None: return 'N/A'
    m = d.get(metric)
    if m is None: return 'N/A'
    f = f"{{:.{decimals}f}}"
    return f"{f.format(m['mean'])} ± {f.format(m['std'])}"

# ── Table row definitions ─────────────────────────────────────────────────────
# Each entry: (metric_label, dataset_label, ds_key, extractor_fn)
ROWS = [
    # Top-1 Accuracy
    ('Top-1 Acc (%)',  'IN-100', 'imagenet100',
     lambda ds, pe: fmt(top1, ds, pe, decimals=2)),
    ('Top-1 Acc (%)',  'C-100',  'cifar100',
     lambda ds, pe: fmt(top1, ds, pe, decimals=2)),

    # Attention Entropy
    ('Attn. Entropy (nats)',  'IN-100', 'imagenet100',
     lambda ds, pe: fmt(entropy, ds, pe, decimals=4)),
    ('Attn. Entropy (nats)',  'C-100',  'cifar100',
     lambda ds, pe: fmt(entropy, ds, pe, decimals=4)),

    # Mutual Information
    ('MI (Pos, Attn) [bits]', 'IN-100', 'imagenet100',
     lambda ds, pe: fmt(mi, ds, pe, decimals=4)),
    ('MI (Pos, Attn) [bits]', 'C-100',  'cifar100',
     lambda ds, pe: fmt(mi, ds, pe, decimals=4)),

    # PE Robustness AUROC
    ('Robustness FGSM',  'IN-100', 'imagenet100',
     lambda ds, pe: fmt_ood(ood, ds, pe, 'fgsm_auroc', 3)),
    ('Robustness FGSM',  'C-100',  'cifar100',
     lambda ds, pe: fmt_ood(ood, ds, pe, 'fgsm_auroc', 3)),
    ('Robustness PGD',   'IN-100', 'imagenet100',
     lambda ds, pe: fmt_ood(ood, ds, pe, 'pgd_auroc',  3)),
    ('Robustness PGD',   'C-100',  'cifar100',
     lambda ds, pe: fmt_ood(ood, ds, pe, 'pgd_auroc',  3)),
    ('Variance Gini',    'IN-100', 'imagenet100',
     lambda ds, pe: fmt_ood(ood, ds, pe, 'variance_gini', 4)),
    ('Variance Gini',    'C-100',  'cifar100',
     lambda ds, pe: fmt_ood(ood, ds, pe, 'variance_gini', 4)),

    # Linear Probe
    ('Linear Probe (%)',  'IN-100', 'imagenet100',
     lambda ds, pe: fmt(probe, ds, pe, 'position', decimals=2)),
    ('Linear Probe (%)',  'C-100',  'cifar100',
     lambda ds, pe: fmt(probe, ds, pe, 'position', decimals=2)),

    # IB Compression Ratio
    ('IB Comp. Ratio ΔH (nats)', 'IN-100', 'imagenet100',
     lambda ds, pe: fmt(ib, ds, pe, decimals=4)),
    ('IB Comp. Ratio ΔH (nats)', 'C-100',  'cifar100',
     lambda ds, pe: fmt(ib, ds, pe, decimals=4)),
]

# ── Build and print table ─────────────────────────────────────────────────────
HDR_METRIC = 28
HDR_DS     = 8
HDR_PE     = 17
SEP        = "-" * (HDR_METRIC + HDR_DS + HDR_PE * 4 + 2)

header = (f"{'Metric':<{HDR_METRIC}} {'DS':<{HDR_DS}}"
          + "".join(f"{PE_LABELS[pe]:>{HDR_PE}}" for pe in PE_TYPES))

lines = [
    "TABLE 2: Comprehensive Benchmark of PE Strategies (μ ± σ, 3 seeds)",
    "=" * len(SEP),
    header,
    SEP,
]
csv_rows  = [['Metric', 'Dataset'] + [PE_LABELS[pe] for pe in PE_TYPES]]
prev_metric = None

for (metric_label, ds_label, ds_key, extractor) in ROWS:
    if prev_metric and prev_metric != metric_label:
        lines.append(SEP)   # visual separator between metrics
    prev_metric = metric_label
    values  = [extractor(ds_key, pe) for pe in PE_TYPES]
    row_str = (f"{metric_label:<{HDR_METRIC}} {ds_label:<{HDR_DS}}"
               + "".join(f"{v:>{HDR_PE}}" for v in values))
    lines.append(row_str)
    csv_rows.append([metric_label, ds_label] + values)

lines += [
    SEP, "",
    "Notes:",
    "  Attn. Entropy  : mid layers 4-8, Shannon entropy of attention distributions (nats)",
    "  MI             : MI(query position, argmax attended key) in last layer (bits)",
    "  Robustness AUROC: area under accuracy-vs-epsilon curve under PE noise (FGSM/PGD)",
    "  Linear Probe   : logistic regression on PE embeddings predicting patch position (%)",
    "  IB Comp. Ratio : H(layers 1-3) - H(layers 10-12) in nats",
    "  Variance Gini  : attention head concentration (1.0 = maximally focused)",
    "  All values     : mean ± std over seeds {42, 123, 456}",
]

full_table = "\n".join(lines)
print(full_table)

# ── Save TXT ──────────────────────────────────────────────────────────────────
txt_path = os.path.join(TABLE_DIR, 'TABLE1_FINAL.txt')
with open(txt_path, 'w') as f:
    f.write(full_table)
print(f"\n✓ TXT saved: {txt_path}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(TABLE_DIR, 'TABLE1_FINAL.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerows(csv_rows)
print(f"✓ CSV saved: {csv_path}")

# ── Coverage check ────────────────────────────────────────────────────────────
print("\n\n── INPUT FILE COVERAGE ──────────────────────────────────────────────────")
for fname, data in [
    ('top1_accuracy.json',     top1),
    ('attention_entropy.json', entropy),
    ('mutual_information.json',mi),
    ('ood_auroc.json',         ood),
    ('linear_probe.json',      probe),
    ('ib_compression.json',    ib),
]:
    status = "✓" if data else "✗ MISSING — run the corresponding script"
    print(f"  {status}  {fname}")

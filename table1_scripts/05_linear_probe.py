"""
SCRIPT 05 — Linear Probe (positional decoding from PE embeddings)
=================================================================

Run in Colab:
    %run /content/05_linear_probe.py

Output: results/table1/linear_probe.json
"""

import os, json, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer, extract_positional_embedding, probe_analysis

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT    = '/content/drive/My Drive/pe_experiment'
RESULTS_IN    = os.path.join(DRIVE_ROOT, 'results')
RESULTS_CF    = os.path.join(DRIVE_ROOT, 'results_cifar100')
ANALYSIS_DATA = os.path.join(RESULTS_IN, 'analysis_data.json')
OUT_DIR       = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_DIR, exist_ok=True)

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS    = [42, 123, 456]
device   = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ── Model loading with architecture auto-detection ────────────────────────────
def load_model(results_root, pe_type, seed, n_classes=100):
    path = os.path.join(results_root, f'{pe_type}_seed{seed}', 'best_model.pth')
    if not os.path.exists(path):
        print(f"  [MISSING] {path}"); return None, None, None

    state      = torch.load(path, map_location=device)
    state      = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    pw         = state['patch_embed.proj.weight']
    patch_size = pw.shape[-1]

    # RoPE/ALiBi may not have a standard pos_embed in state_dict
    pe_shape = None
    for key in ['pos_encoding.pos_embed', 'pos_embed']:
        if key in state:
            pe_shape = state[key].shape; break

    if pe_shape is not None:
        n_tokens  = pe_shape[1]
        candidate = n_tokens - 1
        n_patches = candidate if (candidate > 0 and int(candidate**0.5)**2 == candidate) else n_tokens
        img_size  = int(n_patches**0.5) * patch_size
    else:
        img_size  = 32 if patch_size == 4 else 224
        n_patches = (img_size // patch_size) ** 2

    n_patches_side = int(n_patches**0.5)
    print(f"    [auto] patch_size={patch_size}, img_size={img_size}, "
          f"n_patches={n_patches} ({n_patches_side}x{n_patches_side})")

    torch.manual_seed(seed)
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=n_classes,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, patch_size, n_patches_side

# ── Manual probe with StratifiedKFold ─────────────────────────────────────────
def manual_probe_analysis(pe_matrix: np.ndarray, n_patches_side: int) -> dict:
    """Fit logistic regression probes for row, column, and flat position."""
    n_patches = n_patches_side ** 2

    # Remove CLS token if present
    if pe_matrix.shape[0] == n_patches + 1:
        pe_matrix = pe_matrix[1:]
    if pe_matrix.shape[0] != n_patches:
        print(f"    [WARN] PE shape {pe_matrix.shape[0]} != {n_patches}"); return None

    X         = pe_matrix.astype(np.float32)
    positions = np.arange(n_patches)
    rows      = positions // n_patches_side
    cols      = positions  % n_patches_side

    X_sc = StandardScaler().fit_transform(X)
    results = {}

    for task_name, y in [('row', rows), ('column', cols), ('position', positions)]:
        min_class_size = int(np.min(np.bincount(y)))
        n_splits       = min(5, min_class_size)

        if n_splits < 2:
            # Too few samples per class — fit on full data (upper bound)
            clf  = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', n_jobs=-1)
            clf.fit(X_sc, y)
            accs = [clf.score(X_sc, y) * 100]
            print(f"    probe {task_name}: {accs[0]:.1f}% (no CV, n={len(y)})")
        else:
            skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            accs = []
            for tr, val in skf.split(X_sc, y):
                clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', n_jobs=-1)
                clf.fit(X_sc[tr], y[tr])
                accs.append(clf.score(X_sc[val], y[val]) * 100)
            print(f"    probe {task_name}: {np.mean(accs):.1f}% ± {np.std(accs):.1f}% ({n_splits}-fold)")

        results[task_name] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs))}
    return results

# ── Extract PE matrix from model ──────────────────────────────────────────────
def extract_pe_from_model(model, pe_type: str) -> np.ndarray:
    """Try multiple extraction paths; return PE matrix as numpy array."""
    # 1. Try the helper from full_scale_experiment
    try:
        pe = extract_positional_embedding(model, pe_type)
        if pe is not None:
            arr = pe.detach().cpu().numpy() if isinstance(pe, torch.Tensor) else np.array(pe)
            return arr
    except Exception as e:
        print(f"    [extract_positional_embedding failed] {e}")

    # 2. Search common sub-module attributes
    for attr in ['pos_encoding', 'pe', 'positional_encoding']:
        if hasattr(model, attr):
            obj = getattr(model, attr)
            for sub in ['pos_embed', 'weight', 'embedding']:
                if hasattr(obj, sub):
                    t = getattr(obj, sub)
                    if isinstance(t, torch.Tensor):
                        arr = t.detach().cpu().numpy()
                        if arr.ndim == 3: arr = arr.squeeze(0)
                        print(f"    [PE via model.{attr}.{sub}] shape={arr.shape}")
                        return arr

    # 3. Direct model.pos_embed
    if hasattr(model, 'pos_embed'):
        arr = model.pos_embed.detach().cpu().numpy().squeeze(0)
        print(f"    [PE via model.pos_embed] shape={arr.shape}")
        return arr

    print(f"    [WARN] Cannot extract PE for {pe_type}")
    return None

# 
# PART 1: ImageNet-100 — read directly from analysis_data.json 
# 
print("\n" + "="*55)
print("PART 1: ImageNet-100 — reading from analysis_data.json")
print("="*55)

output = {'imagenet100': {}, 'cifar100': {}}

if os.path.exists(ANALYSIS_DATA):
    with open(ANALYSIS_DATA) as f:
        ad = json.load(f)

    print(f"\n{'PE Type':<14} {'pos μ':>9} {'pos σ':>9}  {'row μ':>9} {'col μ':>9}")
    print("-" * 58)

    for pe in PE_TYPES:
        pos_vals, row_vals, col_vals = [], [], []
        per_seed = {}
        for seed_int in SEEDS:
            s = str(seed_int)
            if s not in ad.get(pe, {}):
                per_seed[seed_int] = None; continue
            probe = ad[pe][s]['probe']
            per_seed[seed_int] = {
                'position': probe['position']['mean'],
                'row':      probe['row']['mean'],
                'column':   probe['column']['mean'],
            }
            pos_vals.append(probe['position']['mean'])
            row_vals.append(probe['row']['mean'])
            col_vals.append(probe['column']['mean'])

        if pos_vals:
            output['imagenet100'][pe] = {
                'per_seed': per_seed,
                'position': {'mean': round(np.mean(pos_vals),4), 'std': round(np.std(pos_vals),4)},
                'row':      {'mean': round(np.mean(row_vals),4), 'std': round(np.std(row_vals),4)},
                'column':   {'mean': round(np.mean(col_vals),4), 'std': round(np.std(col_vals),4)},
                'source':   'analysis_data.json',
            }
            print(f"{pe:<14} {np.mean(pos_vals):9.2f} {np.std(pos_vals):9.4f}  "
                  f"{np.mean(row_vals):9.2f} {np.mean(col_vals):9.2f}")
else:
    print(f"  [MISSING] {ANALYSIS_DATA}")

# 
# PART 2: CIFAR-100 — compute probe directly from model checkpoints 
# 
print("\n" + "="*55)
print("PART 2: CIFAR-100 — computing probe from model checkpoints")
print("="*55)
print(f"\n{'PE Type':<14} {'Seed42':>9} {'Seed123':>9} {'Seed456':>9}  {'μ (pos)':>9} {'σ':>9}")
print("-" * 62)

for pe in PE_TYPES:
    pos_vals, row_vals, col_vals = [], [], []
    per_seed = {}
    n_ps = 8   # default for CIFAR-100 (patch_size=4, img=32 → 8x8)

    for seed in SEEDS:
        print(f"  -> {pe} seed{seed} ...")
        model, patch_size, n_patches_side = load_model(RESULTS_CF, pe, seed)
        if model is None:
            per_seed[seed] = None; continue

        n_ps      = n_patches_side
        pe_matrix = extract_pe_from_model(model, pe)
        probe_res = None

        if pe_matrix is not None:
            try:
                probe_res = probe_analysis(pe_matrix, num_patches_per_side=n_patches_side)
            except Exception as e:
                print(f"    [probe_analysis failed] {e}")
            if probe_res is None:
                print("    [falling back to manual probe analysis]")
                probe_res = manual_probe_analysis(pe_matrix, n_patches_side)

        if probe_res is None:
            per_seed[seed] = None; print("    PROBE FAILED")
        else:
            per_seed[seed] = {
                'position': probe_res['position']['mean'],
                'row':      probe_res['row']['mean'],
                'column':   probe_res['column']['mean'],
            }
            pos_vals.append(probe_res['position']['mean'])
            row_vals.append(probe_res['row']['mean'])
            col_vals.append(probe_res['column']['mean'])
            print(f"    pos={probe_res['position']['mean']:.2f}% OK")

        del model; torch.cuda.empty_cache()

    if pos_vals:
        output['cifar100'][pe] = {
            'per_seed':       per_seed,
            'position':       {'mean': round(np.mean(pos_vals),4), 'std': round(np.std(pos_vals),4)},
            'row':            {'mean': round(np.mean(row_vals),4), 'std': round(np.std(row_vals),4)},
            'column':         {'mean': round(np.mean(col_vals),4), 'std': round(np.std(col_vals),4)},
            'source':         'computed_from_model',
            'n_patches_side': n_ps,
        }
        vals_str = "  ".join(
            f"{per_seed[s]['position']:7.1f}%" if per_seed.get(s) else "    N/A"
            for s in SEEDS
        )
        print(f"{pe:<14} {vals_str}  {np.mean(pos_vals):9.2f} {np.std(pos_vals):9.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'linear_probe.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved: {out_path}")

# ── Print in table format ─────────────────────────────────────────────────────
print("\n\n── TABLE 2 FORMAT ────────────────────────────────────────────────────────")
print(f"{'Metric':<28} {'Dataset':<10} {'Learned':>15} {'Sinusoidal':>15}"
      f" {'RoPE':>15} {'ALiBi':>15}")
print("-" * 90)
for ds_label, ds_key in [('IN-100', 'imagenet100'), ('C-100', 'cifar100')]:
    for task_label, task_key in [
        ('Linear Probe - position (%)', 'position'),
        ('Linear Probe - row (%)',      'row'),
        ('Linear Probe - column (%)',   'column'),
    ]:
        row = f"{task_label:<28} {ds_label:<10}"
        for pe in PE_TYPES:
            d = output.get(ds_key, {}).get(pe)
            if d and task_key in d:
                m    = d[task_key]
                row += f"  {m['mean']:5.2f}±{m['std']:.2f}  "
            else:
                row += f"  {'N/A':^13}  "
        print(row)

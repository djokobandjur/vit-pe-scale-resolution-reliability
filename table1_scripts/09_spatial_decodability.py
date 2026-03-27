"""
SCRIPT 09 — Spatial Decodability Protocol (Table 4)
====================================================
Trains a GPU linear probe to predict the flat patch index from post-block
activations at layers {1, 3, 6, 9, 12}. Computes Spatial Decay = AccL1 - AccL12.

Run in Colab:
    %run /content/09_spatial_decodability.py

Output: results/table1/spatial_decodability.json
"""

import os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT       = '/content/drive/My Drive/pe_experiment'
RESULTS_IN       = os.path.join(DRIVE_ROOT, 'results')
RESULTS_CF       = os.path.join(DRIVE_ROOT, 'results_cifar100')
DATA_IN          = '/content/imagenet100'
OUT_DIR          = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_DIR, exist_ok=True)

PE_TYPES         = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS            = [42, 123, 456]
PROBE_LAYERS     = [1, 3, 6, 9, 12]  # 1-indexed (human-readable)
BATCH_SIZE       = 128
PROBE_EPOCHS     = 10
NUM_PROBE_IMAGES = 2000

IN_MEAN = [0.485, 0.456, 0.406]; IN_STD = [0.229, 0.224, 0.225]
CF_MEAN = [0.5071,0.4867,0.4408]; CF_STD = [0.2675,0.2565,0.2761]

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f"Device: {device} ({gpu_name})")

# ── Data loaders ──────────────────────────────────────────────────────────────
def make_imagenet_loader():
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)
    ])
    ds     = datasets.ImageFolder(os.path.join(DATA_IN, 'val'), tf)
    subset = Subset(ds, list(range(min(NUM_PROBE_IMAGES, len(ds)))))
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)

def make_cifar_loader():
    tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CF_MEAN, CF_STD)
    ])
    ds     = datasets.CIFAR100('/content', train=False, download=True, transform=tf)
    subset = Subset(ds, list(range(min(NUM_PROBE_IMAGES, len(ds)))))
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)

# ── Architecture auto-detection ───────────────────────────────────────────────
def detect_arch(state):
    pw         = state['patch_embed.proj.weight']
    patch_size = pw.shape[-1]
    pe_shape   = None
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
    return patch_size, img_size, n_patches

def load_model(results_root, pe_type, seed):
    path = os.path.join(results_root, f'{pe_type}_seed{seed}', 'best_model.pth')
    if not os.path.exists(path):
        print(f"  [MISSING] {path}"); return None, None
    state      = torch.load(path, map_location=device)
    state      = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    patch_size, img_size, n_patches = detect_arch(state)
    print(f"    [auto] patch_size={patch_size}, img_size={img_size}, n_patches={n_patches}")
    torch.manual_seed(seed)
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=100,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, n_patches

# ── GPU linear probe ──────────────────────────────────────────────────────────
def run_gpu_probe(feats, labels, num_classes, epochs=PROBE_EPOCHS):
    """
    Train a linear head on GPU to predict patch position from activations.
    feats:  [N, D] CPU tensor
    labels: [N]   CPU tensor (flat patch index)
    Returns validation accuracy (%).
    """
    D   = feats.shape[1]
    idx = torch.randperm(len(feats), generator=torch.Generator().manual_seed(42))
    split = int(0.8 * len(feats))
    tr_idx, va_idx = idx[:split], idx[split:]

    # Normalise features using training set statistics
    mean  = feats[tr_idx].mean(0, keepdim=True)
    std   = feats[tr_idx].std(0,  keepdim=True) + 1e-6
    feats = (feats - mean) / std

    tr_X = feats[tr_idx].to(device); tr_y = labels[tr_idx].to(device)
    va_X = feats[va_idx].to(device); va_y = labels[va_idx].to(device)

    head      = nn.Linear(D, num_classes).to(device)
    optimizer = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    tr_loader = DataLoader(TensorDataset(tr_X, tr_y), batch_size=2048, shuffle=True)

    for _ in range(epochs):
        head.train()
        for bx, by in tr_loader:
            optimizer.zero_grad()
            criterion(head(bx), by).backward()
            optimizer.step()

    head.eval()
    with torch.no_grad():
        acc = (head(va_X).argmax(dim=1) == va_y).float().mean().item() * 100
    return round(acc, 4)

# ── Activation extraction via forward hooks ───────────────────────────────────
@torch.no_grad()
def extract_with_hooks(model, loader, probe_layers, n_patches):
    """
    Register forward hooks on model.blocks[l-1] (0-indexed) for each
    1-indexed probe layer. Returns {layer: Tensor [N_img * n_patches, D]}.
    CLS token is excluded; only spatial patch tokens are kept.
    """
    storage = {l: [] for l in probe_layers}
    hooks   = []

    def make_hook(l):
        def hook(module, input, output):
            acts = output[:, 1:, :]         # remove CLS token [B, N, D]
            acts = acts[:, :n_patches, :]   # clip to n_patches
            storage[l].append(acts.cpu())
        return hook

    for l in probe_layers:
        hooks.append(model.blocks[l-1].register_forward_hook(make_hook(l)))

    for imgs, _ in tqdm(loader, desc="  forward", leave=False):
        model(imgs.to(device))

    for h in hooks: h.remove()

    result = {}
    for l in probe_layers:
        if storage[l]:
            cat     = torch.cat(storage[l], dim=0)  # [B_total, N, D]
            B, N, D = cat.shape
            result[l] = cat.reshape(B * N, D)         # [B*N, D]
    return result

# ── Main loop ─────────────────────────────────────────────────────────────────
output = {}

for ds_name, results_root, loader_fn in [
    ('imagenet100', RESULTS_IN, make_imagenet_loader),
    ('cifar100',    RESULTS_CF, make_cifar_loader),
]:
    print(f"\n{'='*60}\nDataset: {ds_name.upper()}\n{'='*60}")
    loader = loader_fn()
    output[ds_name] = {}

    for pe in PE_TYPES:
        print(f"\n── {pe.upper()} ─────────────────────────────────────")
        layer_accs = {l: [] for l in PROBE_LAYERS}
        per_seed   = {}

        for seed in SEEDS:
            print(f"  seed{seed}:")
            model, n_patches = load_model(results_root, pe, seed)
            if model is None:
                per_seed[seed] = None; continue

            acts = extract_with_hooks(model, loader, PROBE_LAYERS, n_patches)
            del model; torch.cuda.empty_cache()

            seed_res = {}
            for l in PROBE_LAYERS:
                if l not in acts:
                    seed_res[l] = float('nan'); continue
                feats  = acts[l]                                # [B*N, D]
                n_act  = feats.shape[0]
                # Labels: flat patch index repeated for each image
                labels = torch.arange(n_patches).repeat(n_act // n_patches)
                labels = labels[:n_act]
                acc    = run_gpu_probe(feats, labels, num_classes=n_patches)
                seed_res[l] = acc
                layer_accs[l].append(acc)
                print(f"    L{l:2d}: {acc:.2f}%")

            # Spatial Decay = L1 accuracy - L12 accuracy
            l1  = seed_res.get(1,  float('nan'))
            l12 = seed_res.get(12, float('nan'))
            seed_res['spatial_decay'] = (
                round(l1 - l12, 4) if not (np.isnan(l1) or np.isnan(l12))
                else float('nan')
            )
            per_seed[seed] = seed_res

        # Aggregate over seeds
        layer_agg = {}
        for l in PROBE_LAYERS:
            vals = layer_accs[l]
            layer_agg[l] = {
                'mean': round(float(np.mean(vals)), 4) if vals else float('nan'),
                'std':  round(float(np.std(vals)),  4) if vals else float('nan'),
            }

        decay_vals = [
            per_seed[s]['spatial_decay'] for s in SEEDS
            if per_seed.get(s) and not np.isnan(per_seed[s].get('spatial_decay', float('nan')))
        ]

        output[ds_name][pe] = {
            'per_seed':    per_seed,
            'layer_probe': layer_agg,
            'spatial_decay': {
                'mean': round(float(np.mean(decay_vals)), 4) if decay_vals else float('nan'),
                'std':  round(float(np.std(decay_vals)),  4) if decay_vals else float('nan'),
            },
        }

        print(f"\n  {pe} summary: "
              + " ".join(f"L{l}={layer_agg[l]['mean']:.1f}%" for l in PROBE_LAYERS))
        print(f"  Spatial Decay: {output[ds_name][pe]['spatial_decay']['mean']:+.2f}%")

    del loader

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'spatial_decodability.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved: {out_path}")

# ── Print Table 4 format ──────────────────────────────────────────────────────
print("\n\n── TABLE 4: SPATIAL DECODABILITY PROTOCOL ─────────────────────────────")
layer_header = "".join(f"{'L'+str(l):>10}" for l in PROBE_LAYERS)
print(f"\n{'Dataset':<10} {'PE Type':<14}{layer_header}  {'Decay':>8}")
print("-" * (10 + 14 + 10 * len(PROBE_LAYERS) + 10))

for ds_label, ds_key in [('IN-100', 'imagenet100'), ('C-100', 'cifar100')]:
    for pe in PE_TYPES:
        d = output.get(ds_key, {}).get(pe)
        if not d:
            print(f"{ds_label:<10} {pe:<14} N/A"); continue
        vals  = "".join(f"{d['layer_probe'][l]['mean']:>8.1f}% " for l in PROBE_LAYERS)
        decay = d['spatial_decay']
        print(f"{ds_label:<10} {pe:<14}{vals} {decay['mean']:+.1f}%")
    print()

print("\n✓ Completed")

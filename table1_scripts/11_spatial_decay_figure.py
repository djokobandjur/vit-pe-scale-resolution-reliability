"""
SCRIPT 11 — Figure 4: Spatial Decodability kroz all 12 layers
=================================================================
Computes linear probe for all layers [1..12] and creates a line chart
with sigma shadow (error bars per 3 seeds).

Figure 4 struktura:
  - X osa: Layer (1-12)
  - Y osa: Position Decoding Accuracy (%)
  - 4 linije: Learned, Sinusoidal, RoPE, ALiBi
  - 2 panela: IN-100 (levo), C-100 (desno)
  - Senka oko linije = +/- sigma kroz 3 seed-a

Output:
  results/figures/figure4_spatial_decay.png (300 DPI)
  results/figures/figure4_spatial_decay.pdf
  results/table1/spatial_decay_all_layers.json
"""

import os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT       = '/content/drive/My Drive/pe_experiment'
RESULTS_IN       = os.path.join(DRIVE_ROOT, 'results')
RESULTS_CF       = os.path.join(DRIVE_ROOT, 'results_cifar100')
DATA_IN          = '/content/imagenet100'
OUT_FIG_DIR      = os.path.join(DRIVE_ROOT, 'results', 'figures')
OUT_DATA_DIR     = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_FIG_DIR,  exist_ok=True)
os.makedirs(OUT_DATA_DIR, exist_ok=True)

PE_TYPES         = ['learned', 'sinusoidal', 'rope', 'alibi']
PE_LABELS        = {'learned': 'Learned', 'sinusoidal': 'Sinusoidal',
                    'rope': 'RoPE', 'alibi': 'ALiBi'}
PE_COLORS        = {'learned': '#E24B4A', 'sinusoidal': '#185FA5',
                    'rope': '#1D9E75', 'alibi': '#BA7517'}
PE_MARKERS       = {'learned': 'o', 'sinusoidal': 's', 'rope': '^', 'alibi': 'D'}

SEEDS            = [42, 123, 456]
ALL_LAYERS       = list(range(1, 13))   # 1..12
BATCH_SIZE       = 128
PROBE_EPOCHS     = 10
NUM_PROBE_IMAGES = 2000

IN_MEAN = [0.485, 0.456, 0.406]; IN_STD = [0.229, 0.224, 0.225]
CF_MEAN = [0.5071,0.4867,0.4408]; CF_STD = [0.2675,0.2565,0.2761]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ── Data loaders ───────────────────────────────────────────────────────────────
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

# ── Architecture auto-detection ────────────────────────────────────────────────
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
    torch.manual_seed(seed)
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=100,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, n_patches

# ── GPU Linear Probe ──────────────────────────────────────────────────────────
def run_gpu_probe(feats, labels, num_classes, epochs=PROBE_EPOCHS):
    D     = feats.shape[1]
    idx   = torch.randperm(len(feats), generator=torch.Generator().manual_seed(42))
    split = int(0.8 * len(feats))
    tr_idx, va_idx = idx[:split], idx[split:]
    mean  = feats[tr_idx].mean(0, keepdim=True)
    std   = feats[tr_idx].std(0,  keepdim=True) + 1e-6
    feats = (feats - mean) / std
    tr_X  = feats[tr_idx].to(device); tr_y = labels[tr_idx].to(device)
    va_X  = feats[va_idx].to(device); va_y = labels[va_idx].to(device)
    head  = nn.Linear(D, num_classes).to(device)
    opt   = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    ldr   = DataLoader(TensorDataset(tr_X, tr_y), batch_size=2048, shuffle=True)
    for _ in range(epochs):
        head.train()
        for bx, by in ldr:
            opt.zero_grad(); crit(head(bx), by).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(va_X).argmax(1) == va_y).float().mean().item() * 100
    return round(acc, 4)

# ── Ekstrakcija all 12 layers odjednom ─────────────────────────────────────
@torch.no_grad()
def extract_all_layers(model, loader, n_patches):
    storage = {l: [] for l in ALL_LAYERS}
    hooks   = []

    def make_hook(l):
        def hook(module, input, output):
            acts = output[:, 1:, :]
            acts = acts[:, :n_patches, :]
            storage[l].append(acts.cpu())
        return hook

    for l in ALL_LAYERS:
        hooks.append(model.blocks[l-1].register_forward_hook(make_hook(l)))

    for imgs, _ in tqdm(loader, desc="  forward", leave=False):
        model(imgs.to(device))

    for h in hooks:
        h.remove()

    result = {}
    for l in ALL_LAYERS:
        if storage[l]:
            cat     = torch.cat(storage[l], dim=0)
            B, N, D = cat.shape
            result[l] = cat.reshape(B * N, D)
    return result

# ── Main loop ─────────────────────────────────────────────────────────────
output = {}

dataset_configs = [
    ('imagenet100', RESULTS_IN, make_imagenet_loader),
    ('cifar100',    RESULTS_CF, make_cifar_loader),
]

for ds_name, results_root, loader_fn in dataset_configs:
    print(f"\n{'='*60}\nDataset: {ds_name.upper()}\n{'='*60}")
    loader = loader_fn()
    output[ds_name] = {}

    for pe in PE_TYPES:
        print(f"\n── {pe.upper()} ─────────────────────────────────────")
        # {layer: [acc_seed42, acc_seed123, acc_seed456]}
        layer_accs = {l: [] for l in ALL_LAYERS}

        for seed in SEEDS:
            print(f"  seed{seed}:", end=' ', flush=True)
            model, n_patches = load_model(results_root, pe, seed)
            if model is None:
                continue

            acts = extract_all_layers(model, loader, n_patches)
            del model; torch.cuda.empty_cache()

            for l in ALL_LAYERS:
                if l not in acts:
                    continue
                feats  = acts[l]
                n_act  = feats.shape[0]
                labels = torch.arange(n_patches).repeat(n_act // n_patches)
                labels = labels[:n_act]
                acc    = run_gpu_probe(feats, labels, num_classes=n_patches)
                layer_accs[l].append(acc)
                print(f"L{l}={acc:.1f}%", end=' ', flush=True)
            print()

        # Aggregate over seeds: mean i std per layer
        layer_agg = {}
        for l in ALL_LAYERS:
            vals = layer_accs[l]
            layer_agg[l] = {
                'mean': round(float(np.mean(vals)), 4) if vals else float('nan'),
                'std':  round(float(np.std(vals)),  4) if vals else float('nan'),
                'vals': vals,
            }

        output[ds_name][pe] = {
            'layer_probe':   layer_agg,
            'spatial_decay': round(
                layer_agg[1]['mean'] - layer_agg[12]['mean'], 4
            ) if layer_agg[1]['mean'] and layer_agg[12]['mean'] else float('nan'),
        }

        # Summary
        means = [layer_agg[l]['mean'] for l in ALL_LAYERS]
        print(f"  Peak: L{ALL_LAYERS[np.argmax(means)]}={max(means):.1f}%  "
              f"L12={layer_agg[12]['mean']:.1f}%  "
              f"Decay={output[ds_name][pe]['spatial_decay']:+.1f}%")

    del loader

# ── Save JSON ──────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DATA_DIR, 'spatial_decay_all_layers.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Podaci sacuvani: {out_path}")

# ── Create Figure 4 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.patch.set_facecolor('white')

ds_configs = [
    ('imagenet100', 'ImageNet-100  (224×224, patch=16, 196 patches)'),
    ('cifar100',    'CIFAR-100  (32×32, patch=4, 64 patches)'),
]

for ax, (ds_key, ds_title) in zip(axes, ds_configs):
    for pe in PE_TYPES:
        d = output.get(ds_key, {}).get(pe)
        if not d:
            continue

        means = np.array([d['layer_probe'][l]['mean'] for l in ALL_LAYERS])
        stds  = np.array([d['layer_probe'][l]['std']  for l in ALL_LAYERS])
        xs    = np.array(ALL_LAYERS)

        # Linija
        ax.plot(xs, means,
                color=PE_COLORS[pe],
                marker=PE_MARKERS[pe],
                markersize=5,
                linewidth=1.8,
                label=PE_LABELS[pe],
                zorder=3)

        # Senka = +/- sigma
        ax.fill_between(xs,
                        means - stds,
                        means + stds,
                        alpha=0.15,
                        color=PE_COLORS[pe],
                        zorder=2)

    # Stilizacija osa
    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel('Position Decoding Accuracy (%)', fontsize=12)
    ax.set_title(ds_title, fontsize=11, pad=10, color='#333333')
    ax.set_xticks(ALL_LAYERS)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Legenda samo na levom panelu
    if ds_key == 'imagenet100':
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right',
                  frameon=True, edgecolor='#dddddd')

    # Vertikalna linija na sloju 3 (best_layer iz cross-domain)
    ax.axvline(x=3, color='#888888', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.text(3.1, 102, 'Best OOD\nlayer (L3)',
            fontsize=7.5, color='#888888', va='top')

# Naslov
fig.suptitle(
    'Spatial Decodability Across Transformer Layers\n'
    'Linear probe accuracy for patch position prediction (shaded area = ±σ, 3 seeds)',
    fontsize=12, y=1.01, color='#1a1a1a'
)

plt.tight_layout()

# Save
png_path = os.path.join(OUT_FIG_DIR, 'figure4_spatial_decay.png')
pdf_path = os.path.join(OUT_FIG_DIR, 'figure4_spatial_decay.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path,           bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"✓ PNG: {png_path}")
print(f"✓ PDF: {pdf_path}")

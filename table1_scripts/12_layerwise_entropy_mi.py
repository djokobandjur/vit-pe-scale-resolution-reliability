"""
SCRIPT 12 — Per-layer Attention Entropy and MI (Figure 1 + Figure 2)
====================================================================
A single forward pass through the model collects:
  1. Attention Entropy per layer [1..12] — for Figure 2 (IB profile)
  2. MI(position, attention) per layer [1..12] — for Figure 1 (MI profile)

Creates two figures:
  Figure 1: MI profile across 12 layers (line chart, shaded = sigma)
  Figure 2: Activation Entropy profile across 12 layers

Output:
  results/figures/figure1_mi_profile.png/pdf
  results/figures/figure2_entropy_profile.png/pdf
  results/table1/layerwise_entropy_mi.json
"""

import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT   = '/content/drive/My Drive/pe_experiment'
RESULTS_IN   = os.path.join(DRIVE_ROOT, 'results')
RESULTS_CF   = os.path.join(DRIVE_ROOT, 'results_cifar100')
DATA_IN      = '/content/imagenet100'
OUT_FIG_DIR  = os.path.join(DRIVE_ROOT, 'results', 'figures')
OUT_DATA_DIR = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_FIG_DIR,  exist_ok=True)
os.makedirs(OUT_DATA_DIR, exist_ok=True)

PE_TYPES   = ['learned', 'sinusoidal', 'rope', 'alibi']
PE_LABELS  = {'learned': 'Learned', 'sinusoidal': 'Sinusoidal',
               'rope': 'RoPE', 'alibi': 'ALiBi'}
PE_COLORS  = {'learned': '#E24B4A', 'sinusoidal': '#185FA5',
               'rope': '#1D9E75', 'alibi': '#BA7517'}
PE_MARKERS = {'learned': 'o', 'sinusoidal': 's', 'rope': '^', 'alibi': 'D'}

SEEDS      = [42, 123, 456]
ALL_LAYERS = list(range(1, 13))
N_BATCHES  = 20
BATCH_SIZE = 64

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
    ds = datasets.ImageFolder(os.path.join(DATA_IN, 'val'), tf)
    return DataLoader(
        Subset(ds, list(range(min(N_BATCHES * BATCH_SIZE, len(ds))))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

def make_cifar_loader():
    tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CF_MEAN, CF_STD)
    ])
    ds = datasets.CIFAR100('/content', train=False, download=True, transform=tf)
    return DataLoader(
        Subset(ds, list(range(min(N_BATCHES * BATCH_SIZE, len(ds))))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
    )

# ── Model utils ───────────────────────────────────────────────────────────────
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

# ── Per-layer ekstrakcija: Entropy i MI u jednom prolasku ─────────────────────
@torch.no_grad()
def compute_layerwise_entropy_and_mi(model, loader, n_patches, n_batches):
    """
    Returns:
      entropy[layer] = float (Shannon entropija attention distribucija, nats)
      mi[layer]      = float (MI pozicija/paznja, bits)
    """
    # Akumulatori
    entropy_vals  = {l: [] for l in ALL_LAYERS}   # per-batch means
    mi_joints     = {l: np.zeros((n_patches, n_patches), dtype=np.int64)
                     for l in ALL_LAYERS}

    # Hookovi za attention weights
    attn_storage = {l: None for l in ALL_LAYERS}
    hooks        = []

    def make_hook(l):
        def hook(module, input, output):
            # output iz block.forward moze biti (x, attn_w) ili samo x
            # Koristimo return_attention=True u forward pozivu
            pass  # hookovcemo direktno u forward petlji
        return hook

    for batch_idx, (imgs, _) in enumerate(tqdm(loader, total=n_batches, leave=False)):
        if batch_idx >= n_batches: break
        imgs = imgs.to(device)
        B    = imgs.shape[0]

        # Manuelni forward
        x = model.patch_embed(imgs)
        if hasattr(model, 'cls_token'):
            x = torch.cat([model.cls_token.expand(B, -1, -1), x], dim=1)
        x = model.pos_encoding(x) if hasattr(model, 'pos_encoding') else x
        if hasattr(model, 'dropout'):
            x = model.dropout(x)

        for i, block in enumerate(model.blocks):
            layer_human = i + 1   # 1-indeksirani

            try:
                out = block(x, return_attention=True)
                if isinstance(out, tuple):
                    x, attn_w = out
                else:
                    x    = out
                    attn_w = None
            except TypeError:
                x      = block(x)
                attn_w = None

            if attn_w is None or layer_human not in ALL_LAYERS:
                continue

            # attn_w: [B, H, N, N] ili [B, N, N]
            if attn_w.dim() == 4:
                attn_w = attn_w.mean(dim=1)   # avg po glavama [B, N, N]

            # ── Attention Entropy ──────────────────────────────────────────
            eps = 1e-9
            H   = -(attn_w * (attn_w + eps).log()).sum(dim=-1).mean().item()
            entropy_vals[layer_human].append(H)

            # ── MI (pozicija, paznja) ──────────────────────────────────────
            # Ukloni CLS ako postoji
            if attn_w.shape[1] == n_patches + 1:
                attn_w = attn_w[:, 1:, 1:]
            attn_w  = attn_w[:, :n_patches, :n_patches]
            attn_np = attn_w.cpu().numpy()

            for b in range(attn_np.shape[0]):
                kmax = attn_np[b].argmax(axis=1)
                for q in range(n_patches):
                    mi_joints[layer_human][q, kmax[q]] += 1

    # Aggregate over seeds
    def entropy_from_joint(joint):
        total = joint.sum()
        if total == 0: return float('nan')
        pxy = joint / total
        px  = pxy.sum(axis=1); py = pxy.sum(axis=0)
        def H(p): p = p[p > 0]; return float(-np.sum(p * np.log2(p)))
        hxy = -np.sum(pxy[pxy > 0] * np.log2(pxy[pxy > 0]))
        return float(max(H(px) + H(py) - hxy, 0.0))

    result_entropy = {}
    result_mi      = {}
    for l in ALL_LAYERS:
        result_entropy[l] = float(np.mean(entropy_vals[l])) if entropy_vals[l] else float('nan')
        result_mi[l]      = entropy_from_joint(mi_joints[l])

    return result_entropy, result_mi

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

        # Per-layer akumulatori kroz seed-ove
        all_entropy = {l: [] for l in ALL_LAYERS}
        all_mi      = {l: [] for l in ALL_LAYERS}

        for seed in SEEDS:
            print(f"  seed{seed} ...", end=' ', flush=True)
            model, n_patches = load_model(results_root, pe, seed)
            if model is None:
                print("SKIP"); continue

            ent, mi = compute_layerwise_entropy_and_mi(
                model, loader, n_patches, N_BATCHES
            )
            del model; torch.cuda.empty_cache()

            for l in ALL_LAYERS:
                if not np.isnan(ent[l]): all_entropy[l].append(ent[l])
                if not np.isnan(mi[l]):  all_mi[l].append(mi[l])

            print(f"done (ent L6={ent[6]:.3f}, mi L6={mi[6]:.3f})")

        # Aggregate over seeds
        entropy_agg = {}
        mi_agg      = {}
        for l in ALL_LAYERS:
            entropy_agg[l] = {
                'mean': round(float(np.mean(all_entropy[l])), 4) if all_entropy[l] else float('nan'),
                'std':  round(float(np.std(all_entropy[l])),  4) if all_entropy[l] else float('nan'),
            }
            mi_agg[l] = {
                'mean': round(float(np.mean(all_mi[l])), 4) if all_mi[l] else float('nan'),
                'std':  round(float(np.std(all_mi[l])),  4) if all_mi[l] else float('nan'),
            }

        output[ds_name][pe] = {
            'attention_entropy': entropy_agg,
            'mutual_information': mi_agg,
        }

        # Summary
        ent_means = [entropy_agg[l]['mean'] for l in ALL_LAYERS]
        mi_means  = [mi_agg[l]['mean']      for l in ALL_LAYERS]
        print(f"  Entropy: L1={ent_means[0]:.3f} → L12={ent_means[11]:.3f}")
        print(f"  MI:      L1={mi_means[0]:.3f} → L12={mi_means[11]:.3f}")

    del loader

# ── Save JSON ──────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DATA_DIR, 'layerwise_entropy_mi.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Podaci sacuvani: {out_path}")

# ── Pomocna funkcija za plotovanje ────────────────────────────────────────────
def plot_layerwise(output, metric_key, ylabel, title, fname, ylim=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    ds_configs = [
        ('imagenet100', 'ImageNet-100  (224×224, patch=16)'),
        ('cifar100',    'CIFAR-100  (32×32, patch=4)'),
    ]

    for ax, (ds_key, ds_title) in zip(axes, ds_configs):
        for pe in PE_TYPES:
            d = output.get(ds_key, {}).get(pe, {}).get(metric_key, {})
            if not d: continue

            means = np.array([d[l]['mean'] for l in ALL_LAYERS])
            stds  = np.array([d[l]['std']  for l in ALL_LAYERS])
            xs    = np.array(ALL_LAYERS)

            ax.plot(xs, means,
                    color=PE_COLORS[pe], marker=PE_MARKERS[pe],
                    markersize=5, linewidth=1.8,
                    label=PE_LABELS[pe], zorder=3)
            ax.fill_between(xs, means - stds, means + stds,
                            alpha=0.15, color=PE_COLORS[pe], zorder=2)

        ax.set_xlabel('Transformer Layer', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ds_title, fontsize=11, pad=10, color='#333333')
        ax.set_xticks(ALL_LAYERS)
        ax.set_xlim(0.5, 12.5)
        if ylim: ax.set_ylim(ylim)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ds_key == 'imagenet100':
            ax.legend(fontsize=10, framealpha=0.9, loc='best',
                      frameon=True, edgecolor='#dddddd')

        # Oznaka mid-layers regiona
        ax.axvspan(4, 8, alpha=0.05, color='gray')
        ax.text(6, ax.get_ylim()[1] * 0.98, 'mid layers',
                fontsize=8, color='#aaaaaa', ha='center', va='top')

    fig.suptitle(title, fontsize=12, y=1.01, color='#1a1a1a')
    plt.tight_layout()

    png = os.path.join(OUT_FIG_DIR, f'{fname}.png')
    pdf = os.path.join(OUT_FIG_DIR, f'{fname}.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf,           bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {fname}.png/pdf")

# ── Figure 1: MI profil ───────────────────────────────────────────────────────
plot_layerwise(
    output,
    metric_key = 'mutual_information',
    ylabel     = 'MI (Position, Attention) [bits]',
    title      = 'Figure 1: Mutual Information Profile Across Layers\n'
                 'MI between query position and attended key position (shaded = ±σ, 3 seeds)',
    fname      = 'figure1_mi_profile',
)

# ── Figure 2: Attention Entropy profil ───────────────────────────────────────
plot_layerwise(
    output,
    metric_key = 'attention_entropy',
    ylabel     = 'Attention Entropy (nats)',
    title      = 'Figure 2: Attention Entropy Profile Across Layers\n'
                 'Shannon entropy of attention distributions (shaded = ±σ, 3 seeds)',
    fname      = 'figure2_entropy_profile',
)

print("\nCompleted! Figure 1 i 2 saved in results/figures/")

"""
SCRIPT 06 — IB Compression Ratio (Delta Activation Entropy)
============================================================

Run in Colab:
    %run /content/06_ib_compression.py

Output: results/table1/ib_compression.json
"""

import os, json, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT   = '/content/drive/My Drive/pe_experiment'
RESULTS_IN   = os.path.join(DRIVE_ROOT, 'results')
RESULTS_CF   = os.path.join(DRIVE_ROOT, 'results_cifar100')
DATA_IN      = '/content/imagenet100'
OUT_DIR      = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_DIR, exist_ok=True)

PE_TYPES     = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS        = [42, 123, 456]
EARLY_LAYERS = [0, 1, 2]    # layers 1-3 (0-indexed)
LATE_LAYERS  = [9, 10, 11]  # layers 10-12 (0-indexed)
N_BATCHES    = 15
BATCH_SIZE   = 64
N_BINS       = 50

IN_MEAN = [0.485, 0.456, 0.406]; IN_STD = [0.229, 0.224, 0.225]
CF_MEAN = [0.5071,0.4867,0.4408]; CF_STD = [0.2675,0.2565,0.2761]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ── Data loaders ──────────────────────────────────────────────────────────────
def make_imagenet_loader():
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)
    ])
    return DataLoader(
        datasets.ImageFolder(os.path.join(DATA_IN, 'val'), tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

def make_cifar_loader():
    # CIFAR fed to CIFAR-trained models — no resize needed
    tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CF_MEAN, CF_STD)
    ])
    return DataLoader(
        datasets.CIFAR100('/content', train=False, download=True, transform=tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

# ── Architecture auto-detection ───────────────────────────────────────────────
def detect_arch(state: dict):
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
        img_size = 32 if patch_size == 4 else 224
    return patch_size, img_size

def load_model(results_root, pe_type, seed, n_classes=100):
    path = os.path.join(results_root, f'{pe_type}_seed{seed}', 'best_model.pth')
    if not os.path.exists(path):
        print(f"  [MISSING] {path}"); return None
    state      = torch.load(path, map_location=device)
    state      = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    patch_size, img_size = detect_arch(state)
    print(f"    [auto] patch_size={patch_size}, img_size={img_size}")
    torch.manual_seed(seed)
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=n_classes,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ── Activation collector (forward hooks) ─────────────────────────────────────
class ActivationCollector:
    """Register forward hooks on LayerNorm outputs to collect activations."""
    def __init__(self, model, layer_indices):
        self.data  = {i: [] for i in layer_indices}
        self.hooks = []
        for i in layer_indices:
            h = model.blocks[i].norm1.register_forward_hook(self._make_hook(i))
            self.hooks.append(h)

    def _make_hook(self, idx):
        def hook(module, input, output):
            # Exclude CLS token (index 0), keep spatial tokens only
            self.data[idx].append(output[:, 1:, :].detach().cpu())
        return hook

    def remove(self):
        for h in self.hooks: h.remove()

    def get_all(self, idx):
        """Return all collected activations for layer idx as [N, D] array."""
        batches = self.data[idx]
        if not batches: return None
        cat = torch.cat(batches, dim=0)
        return cat.reshape(-1, cat.shape[-1]).numpy()

# ── Entropy of activation distribution ───────────────────────────────────────
def activation_entropy(acts: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Estimate activation entropy by histogramming a random subset of dimensions.
    Returns mean entropy across sampled dimensions (in nats).
    """
    D           = acts.shape[1]
    sample_dims = np.random.choice(D, size=min(D, 256), replace=False)
    entropies   = []
    for d in sample_dims:
        hist, _ = np.histogram(acts[:, d], bins=n_bins)
        hist     = hist[hist > 0].astype(float)
        p        = hist / hist.sum()
        entropies.append(float(-np.sum(p * np.log(p))))
    return float(np.mean(entropies))

@torch.no_grad()
def compute_ib_ratio(model, loader, n_batches):
    """
    Run N_BATCHES forward passes collecting activations via hooks,
    then compute ΔH = H_early - H_late.
    """
    all_layers = sorted(set(EARLY_LAYERS + LATE_LAYERS))
    collector  = ActivationCollector(model, all_layers)

    for i, (imgs, _) in enumerate(tqdm(loader, total=n_batches, leave=False)):
        if i >= n_batches: break
        model(imgs.to(device))

    collector.remove()

    layer_entropy = {}
    for idx in all_layers:
        acts = collector.get_all(idx)
        if acts is not None:
            layer_entropy[idx] = activation_entropy(acts)

    if not layer_entropy: return float('nan'), {}

    h_early = np.mean([layer_entropy[i] for i in EARLY_LAYERS if i in layer_entropy])
    h_late  = np.mean([layer_entropy[i] for i in LATE_LAYERS  if i in layer_entropy])
    return float(h_early - h_late), layer_entropy

# ── Main loop ─────────────────────────────────────────────────────────────────
output  = {}
configs = [
    ('imagenet100', RESULTS_IN, make_imagenet_loader, 100),
    ('cifar100',    RESULTS_CF, make_cifar_loader,    100),
]

for ds_name, results_root, loader_fn, n_classes in configs:
    print(f"\n{'='*58}\nDataset: {ds_name.upper()}\n{'='*58}")
    loader = loader_fn()
    output[ds_name] = {}
    print(f"\n{'PE Type':<14} {'Seed42':>10} {'Seed123':>10} {'Seed456':>10}  {'μ':>8} {'σ':>8}")
    print("-" * 68)

    for pe in PE_TYPES:
        vals, per_seed = [], {}
        for seed in SEEDS:
            print(f"  -> {pe} seed{seed} ...", end=' ', flush=True)
            model = load_model(results_root, pe, seed, n_classes)
            if model is None:
                per_seed[seed] = None; print("SKIP"); continue
            ib_ratio, layer_ents = compute_ib_ratio(model, loader, N_BATCHES)
            per_seed[seed] = {
                'ib_ratio':      round(ib_ratio, 4),
                'layer_entropy': {str(k): round(v, 4) for k, v in layer_ents.items()},
            }
            vals.append(ib_ratio)
            print(f"ΔH={ib_ratio:.4f} nats")
            del model; torch.cuda.empty_cache()

        if vals:
            mu, sigma = np.mean(vals), np.std(vals)
            output[ds_name][pe] = {
                'per_seed':     per_seed,
                'mean':         round(mu, 4),
                'std':          round(sigma, 4),
                'unit':         'nats',
                'early_layers': EARLY_LAYERS,
                'late_layers':  LATE_LAYERS,
            }
            seeds_str = "  ".join(
                f"{per_seed[s]['ib_ratio']:8.4f}" if per_seed.get(s) else "     N/A"
                for s in SEEDS
            )
            print(f"{pe:<14} {seeds_str}  {mu:8.4f} {sigma:8.4f}")
    del loader

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'ib_compression.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved: {out_path}")

# ── Print in table format ─────────────────────────────────────────────────────
print("\n\n── TABLE 2 FORMAT ────────────────────────────────────────────────────────")
print(f"{'Metric':<28} {'Dataset':<10} {'Learned':>15} {'Sinusoidal':>15}"
      f" {'RoPE':>15} {'ALiBi':>15}")
print("-" * 90)
for ds_label, ds_key in [('IN-100', 'imagenet100'), ('C-100', 'cifar100')]:
    row = f"{'IB Comp. Ratio ΔH (nats)':<28} {ds_label:<10}"
    for pe in PE_TYPES:
        d    = output.get(ds_key, {}).get(pe)
        row += f"  {d['mean']:.4f}±{d['std']:.4f}  " if d else f"  {'N/A':^13}  "
    print(row)
print(f"\n  (early layers={EARLY_LAYERS}, late layers={LATE_LAYERS}, N_bins={N_BINS})")

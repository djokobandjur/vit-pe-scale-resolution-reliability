"""
SCRIPT 03 — Mutual Information MI(Position, Attention) in the last layer
=========================================================================
Computes MI between query-token position and the argmax-attended key position,
estimated via a joint histogram over N_BATCHES mini-batches.

Run in Colab:
    %run /content/03_mutual_information.py

Output: results/table1/mutual_information.json
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
DRIVE_ROOT = '/content/drive/My Drive/pe_experiment'
RESULTS_IN = os.path.join(DRIVE_ROOT, 'results')
RESULTS_CF = os.path.join(DRIVE_ROOT, 'results_cifar100')
DATA_IN    = '/content/imagenet100'
OUT_DIR    = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_DIR, exist_ok=True)

PE_TYPES   = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS      = [42, 123, 456]
N_BATCHES  = 20
BATCH_SIZE = 128

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
        img_size  = 32 if patch_size == 4 else 224
        n_patches = (img_size // patch_size) ** 2
    return patch_size, img_size, n_patches

def load_model(results_root, pe_type, seed, n_classes=100):
    path = os.path.join(results_root, f'{pe_type}_seed{seed}', 'best_model.pth')
    if not os.path.exists(path):
        print(f"  [MISSING] {path}"); return None, 64
    state      = torch.load(path, map_location=device)
    state      = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    patch_size, img_size, n_patches = detect_arch(state)
    print(f"    [auto] patch_size={patch_size}, img_size={img_size}, n_patches={n_patches}")
    torch.manual_seed(seed)
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=n_classes,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, n_patches

# ── MI computation ────────────────────────────────────────────────────────────
def entropy_from_counts(counts):
    """Shannon entropy (bits) from integer count array."""
    p = counts / counts.sum(); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

@torch.no_grad()
def compute_mi_last_layer(model, loader, n_batches, n_patches):
    """
    Manually forward through all blocks, requesting attention weights
    from the last block only.
    Builds a joint histogram of (query_position, argmax_attended_key_position)
    and computes MI in bits.
    """
    joint    = np.zeros((n_patches, n_patches), dtype=np.int64)
    got_attn = False

    for batch_idx, (imgs, _) in enumerate(tqdm(loader, total=n_batches, leave=False)):
        if batch_idx >= n_batches:
            break
        imgs = imgs.to(device)

        # Manual forward: patch_embed → CLS concat → pos_encoding → blocks
        B = imgs.shape[0]
        x = model.patch_embed(imgs)
        if hasattr(model, 'cls_token'):
            x = torch.cat([model.cls_token.expand(B, -1, -1), x], dim=1)
        if hasattr(model, 'pos_encoding'):
            x = model.pos_encoding(x)
        if hasattr(model, 'dropout'):
            x = model.dropout(x)

        # All blocks except the last — no attention weights needed
        for block in model.blocks[:-1]:
            try:
                out = block(x, return_attention=False)
                x   = out[0] if isinstance(out, tuple) else out
            except TypeError:
                x = block(x)

        # Last block — request attention weights
        attn_w = None
        try:
            out = model.blocks[-1](x, return_attention=True)
            if isinstance(out, tuple):
                x, attn_w = out
            else:
                x = out
        except TypeError:
            x = model.blocks[-1](x)

        if attn_w is None:
            continue
        got_attn = True

        if attn_w.dim() == 4:
            attn_w = attn_w.mean(dim=1)           # average over heads [B, N, N]
        if attn_w.shape[1] == n_patches + 1:
            attn_w = attn_w[:, 1:, 1:]             # remove CLS token
        attn_w  = attn_w[:, :n_patches, :n_patches]
        attn_np = attn_w.cpu().numpy()

        for b in range(attn_np.shape[0]):
            kmax = attn_np[b].argmax(axis=1)        # argmax attended key per query
            for q in range(n_patches):
                joint[q, kmax[q]] += 1

    if not got_attn:
        print("    [WARN] No attention weights captured!")
        return float('nan')

    total = joint.sum()
    if total == 0: return float('nan')

    pxy = joint / total
    px  = pxy.sum(axis=1); py = pxy.sum(axis=0)
    hx  = entropy_from_counts((px * total).astype(np.int64))
    hy  = entropy_from_counts((py * total).astype(np.int64))
    hxy = -np.sum(pxy[pxy > 0] * np.log2(pxy[pxy > 0]))
    return float(max(hx + hy - hxy, 0.0))

# ── Main loop ─────────────────────────────────────────────────────────────────
output = {}
configs = [
    ('imagenet100', RESULTS_IN, make_imagenet_loader, 100),
    ('cifar100',    RESULTS_CF, make_cifar_loader,    100),
]

for ds_name, results_root, loader_fn, n_classes in configs:
    print(f"\n{'='*55}\nDataset: {ds_name.upper()}\n{'='*55}")
    loader = loader_fn()
    output[ds_name] = {}
    print(f"\n{'PE Type':<14} {'Seed42':>9} {'Seed123':>9} {'Seed456':>9}  {'μ':>8} {'σ':>8}")
    print("-" * 68)

    for pe in PE_TYPES:
        vals, per_seed = [], {}
        for seed in SEEDS:
            print(f"  -> {pe} seed{seed} ...", end=' ', flush=True)
            model, n_patches = load_model(results_root, pe, seed, n_classes)
            if model is None:
                per_seed[seed] = None; print("SKIP"); continue
            mi = compute_mi_last_layer(model, loader, N_BATCHES, n_patches)
            per_seed[seed] = round(mi, 4) if not np.isnan(mi) else None
            if not np.isnan(mi):
                vals.append(mi); print(f"{mi:.4f} bits")
            else:
                print("FAILED")
            del model; torch.cuda.empty_cache()

        if vals:
            mu, sigma = np.mean(vals), np.std(vals)
            output[ds_name][pe] = {
                'per_seed': per_seed, 'mean': round(mu,4), 'std': round(sigma,4),
                'unit': 'bits', 'n_patches': n_patches,
            }
            seeds_str = "  ".join(
                f"{per_seed[s]:7.4f}" if per_seed.get(s) else "    N/A"
                for s in SEEDS
            )
            print(f"{pe:<14} {seeds_str}  {mu:8.4f} {sigma:8.4f}")
    del loader

# ── Save — merge with existing if IN-100 results already present ──────────────
out_path = os.path.join(OUT_DIR, 'mutual_information.json')
if os.path.exists(out_path):
    with open(out_path) as f:
        existing = json.load(f)
    existing.update(output)
    output = existing

with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved: {out_path}")

# ── Print in table format ─────────────────────────────────────────────────────
print("\n\n── TABLE 2 FORMAT ────────────────────────────────────────────────────────")
print(f"{'Metric':<28} {'Dataset':<10} {'Learned':>15} {'Sinusoidal':>15} {'RoPE':>15} {'ALiBi':>15}")
print("-" * 90)
for ds_label, ds_key in [('IN-100', 'imagenet100'), ('C-100', 'cifar100')]:
    row = f"{'MI (Pos, Attn) [bits]':<28} {ds_label:<10}"
    for pe in PE_TYPES:
        d    = output.get(ds_key, {}).get(pe)
        row += f"  {d['mean']:.4f}±{d['std']:.4f}  " if d else f"  {'N/A':^13}  "
    print(row)

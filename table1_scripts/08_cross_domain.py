"""
SCRIPT 08 — Cross-Domain Evaluation (Table 3)
==============================================
Direction A: IN-100 models evaluated on C-100 images (Semantic Degradation)
Direction B: C-100 models evaluated on IN-100 images (Structural Pressure)

Metrics computed:
  1. Attention Entropy Shift (Delta H) — per layer and averaged over mid-layers
  2. OOD AUROC per layer               — AUROC at each of the 12 layers individually
  3. MI Stability                      — MI(position, attention) on cross-domain images

Key design decisions:
  - AUROC computed per-layer to find the best discriminating layer
  - Attention Entropy used as uncertainty score (not Softmax confidence)
  - Per-sample entropy used for AUROC (not batch mean)
  - Overconfidence flag raised when mid-layer AUROC < 0.6

Output:
  results/table1/cross_domain.json

Run in Colab:
  %run /content/08_cross_domain.py
"""

import os, json, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT  = '/content/drive/My Drive/pe_experiment'
RESULTS_IN  = os.path.join(DRIVE_ROOT, 'results')
RESULTS_CF  = os.path.join(DRIVE_ROOT, 'results_cifar100')
DATA_IN     = '/content/imagenet100'
OUT_DIR     = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_DIR, exist_ok=True)

PE_TYPES   = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS      = [42, 123, 456]
N_BATCHES  = 20
BATCH_SIZE = 64
N_LAYERS   = 12
MID_LAYERS = list(range(3, 8))  # 0-indexed layers 3-7 = human layers 4-8

IN_MEAN = [0.485, 0.456, 0.406]; IN_STD = [0.229, 0.224, 0.225]
CF_MEAN = [0.5071,0.4867,0.4408]; CF_STD = [0.2675,0.2565,0.2761]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ── Transforms ────────────────────────────────────────────────────────────────
# Direction A: C-100 images resized to 224x224 with IN normalisation
tf_cf_to_224   = transforms.Compose([
    transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize(IN_MEAN, IN_STD)])
# Direction A: IN-100 in-distribution (standard val preprocessing)
tf_in_standard = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)])
# Direction B: IN-100 images downscaled to 32x32 with CF normalisation
tf_in_to_32    = transforms.Compose([
    transforms.Resize((32, 32)),  # force square — Resize(32) only scales shorter side
    transforms.ToTensor(),
    transforms.Normalize(CF_MEAN, CF_STD)])
# Direction B: C-100 in-distribution (standard val preprocessing)
tf_cf_standard = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(CF_MEAN, CF_STD)])

# ── DataLoader factory ────────────────────────────────────────────────────────
def make_loader(dataset, n_batches=N_BATCHES, num_workers=4):
    """
    FIX: called inside each seed loop to create a fresh iterator.
    FIX: num_workers=0 for CIFAR Subset to prevent Colab worker hang.
    """
    n      = min(len(dataset), n_batches * BATCH_SIZE)
    subset = Subset(dataset, list(range(n)))
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=num_workers,
                      pin_memory=(num_workers > 0))

def get_datasets_a():
    ds_in  = datasets.ImageFolder(os.path.join(DATA_IN, 'val'), tf_in_standard)
    ds_ood = datasets.CIFAR100('/content', train=False, download=True,
                                transform=tf_cf_to_224)
    return ds_in, ds_ood

def get_datasets_b():
    ds_in  = datasets.CIFAR100('/content', train=False, download=True,
                                transform=tf_cf_standard)
    ds_ood = datasets.ImageFolder(os.path.join(DATA_IN, 'val'), tf_in_to_32)
    return ds_in, ds_ood

# ── Model utilities ───────────────────────────────────────────────────────────
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

# ── Manual forward pass ───────────────────────────────────────────────────────
def forward_to_blocks(model, imgs):
    """
    Patch embed + CLS token + positional encoding.
    FIX: handles RoPE/ALiBi which have no pos_encoding attribute.
    """
    B = imgs.shape[0]
    x = model.patch_embed(imgs)
    if hasattr(model, 'cls_token'):
        x = torch.cat([model.cls_token.expand(B, -1, -1), x], dim=1)
    if hasattr(model, 'pos_encoding'):
        x = model.pos_encoding(x)
    if hasattr(model, 'dropout'):
        x = model.dropout(x)
    return x

# ── Per-sample, per-layer entropy ─────────────────────────────────────────────
@torch.no_grad()
def get_per_sample_entropy(model, loader, n_batches):
    """
    Returns {layer_idx (0-based): np.array of per-sample entropy values}.
    Per-sample entropy (not batch mean) is required for meaningful AUROC.
    """
    all_entropies = {i: [] for i in range(N_LAYERS)}

    for batch_idx, (imgs, _) in enumerate(tqdm(loader, total=n_batches, leave=False)):
        if batch_idx >= n_batches: break
        imgs = imgs.to(device)
        x    = forward_to_blocks(model, imgs)

        for i, block in enumerate(model.blocks):
            try:
                out = block(x, return_attention=True)
                if isinstance(out, tuple):
                    x, attn_w = out
                    if attn_w is not None:
                        if attn_w.dim() == 4:
                            attn_w = attn_w.mean(dim=1)  # average over heads [B,N,N]
                        eps = 1e-9
                        H   = -(attn_w * (attn_w + eps).log()).sum(dim=-1).mean(dim=-1)
                        all_entropies[i].extend(H.cpu().numpy().tolist())
                else:
                    x = out
            except TypeError:
                x = block(x)

    return {i: np.array(v) for i, v in all_entropies.items() if len(v) > 0}

# ── Layer-wise AUROC ──────────────────────────────────────────────────────────
def compute_layerwise_auroc(ent_in, ent_ood):
    """
    Compute OOD detection AUROC at each layer.
    Label convention: in-distribution=0, OOD=1.
    Higher entropy → higher OOD score.
    """
    aurocs        = {}
    common_layers = set(ent_in.keys()) & set(ent_ood.keys())
    for layer in sorted(common_layers):
        s_in  = ent_in[layer]
        s_ood = ent_ood[layer]
        if len(s_in) == 0 or len(s_ood) == 0: continue
        labels = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_ood))])
        scores = np.concatenate([s_in, s_ood])
        try:
            aurocs[layer] = float(roc_auc_score(labels, scores))
        except Exception:
            aurocs[layer] = float('nan')
    return aurocs

# ── MI on cross-domain images ─────────────────────────────────────────────────
@torch.no_grad()
def compute_mi(model, loader, n_batches, n_patches):
    """Compute MI(position, attention) on OOD images using the last block."""
    joint = np.zeros((n_patches, n_patches), dtype=np.int64)
    got   = False

    for batch_idx, (imgs, _) in enumerate(tqdm(loader, total=n_batches, leave=False)):
        if batch_idx >= n_batches: break
        imgs = imgs.to(device)
        x    = forward_to_blocks(model, imgs)

        for block in model.blocks[:-1]:
            try:
                out = block(x, return_attention=False)
                x   = out[0] if isinstance(out, tuple) else out
            except TypeError:
                x = block(x)

        try:
            out = model.blocks[-1](x, return_attention=True)
            if isinstance(out, tuple):
                x, attn_w = out
                if attn_w is not None:
                    got = True
                    if attn_w.dim() == 4: attn_w = attn_w.mean(dim=1)
                    if attn_w.shape[1] == n_patches + 1:
                        attn_w = attn_w[:, 1:, 1:]
                    attn_w = attn_w[:, :n_patches, :n_patches].cpu().numpy()
                    for b in range(attn_w.shape[0]):
                        kmax = attn_w[b].argmax(axis=1)
                        for q in range(n_patches): joint[q, kmax[q]] += 1
        except TypeError: pass

    if not got or joint.sum() == 0: return float('nan')
    pxy = joint / joint.sum()
    px  = pxy.sum(axis=1); py = pxy.sum(axis=0)
    def H(p): p = p[p > 0]; return float(-np.sum(p * np.log2(p)))
    hxy = -np.sum(pxy[pxy > 0] * np.log2(pxy[pxy > 0]))
    return float(max(H(px) + H(py) - hxy, 0.0))

# ── Process one direction ─────────────────────────────────────────────────────
def process_direction(direction_name, results_root, ds_in, ds_ood,
                      num_workers_in=4, num_workers_ood=0):
    """
    FIX: accepts Dataset objects, creates fresh DataLoaders per seed.
    FIX: num_workers_ood=0 by default to prevent Direction B worker hang.
    """
    print(f"\n{'='*60}\n{direction_name}\n{'='*60}")

    direction_results = {}

    for pe in PE_TYPES:
        print(f"\n── {pe.upper()} ─────────────────────────────────────")
        per_seed        = {}
        all_layer_auroc = {i: [] for i in range(N_LAYERS)}
        delta_mids, mis = [], []

        for seed in SEEDS:
            print(f"  seed{seed} ...", end=' ', flush=True)
            model, n_patches = load_model(results_root, pe, seed)
            if model is None:
                per_seed[seed] = None; print("SKIP"); continue

            # FIX: fresh loaders for each seed
            loader_in  = make_loader(ds_in,  num_workers=num_workers_in)
            loader_ood = make_loader(ds_ood, num_workers=num_workers_ood)

            ent_in       = get_per_sample_entropy(model, loader_in,  N_BATCHES)
            ent_ood      = get_per_sample_entropy(model, loader_ood, N_BATCHES)
            layer_aurocs = compute_layerwise_auroc(ent_in, ent_ood)

            mid_in    = np.mean([ent_in[l].mean()  for l in MID_LAYERS if l in ent_in])
            mid_ood   = np.mean([ent_ood[l].mean() for l in MID_LAYERS if l in ent_ood])
            delta     = float(mid_ood - mid_in)
            best_layer= max(layer_aurocs, key=layer_aurocs.get) if layer_aurocs else None
            best_auroc= layer_aurocs.get(best_layer, float('nan'))
            mid_auroc = float(np.mean([layer_aurocs[l] for l in MID_LAYERS
                                        if l in layer_aurocs]))
            overconfident = mid_auroc < 0.6
            mi        = compute_mi(model, loader_ood, N_BATCHES, n_patches)

            # FIX: store with 1-based layer index for human readability
            layer_aurocs_1idx = {str(k+1): round(v,4) for k,v in layer_aurocs.items()}

            per_seed[seed] = {
                'delta_entropy': round(delta, 4),
                'mid_auroc':     round(mid_auroc, 4),
                'best_auroc':    round(best_auroc, 4),
                'best_layer':    f'L{best_layer+1}' if best_layer is not None else None,
                'overconfident': overconfident,
                'mi_ood':        round(mi, 4),
                'layer_aurocs':  layer_aurocs_1idx,
            }
            delta_mids.append(delta)
            mis.append(mi)
            for l, a in layer_aurocs.items():
                if not np.isnan(a): all_layer_auroc[l].append(a)

            flag = " [OVERCONFIDENT]" if overconfident else ""
            print(f"ΔH={delta:+.4f}  mid_AUROC={mid_auroc:.4f}"
                  f"  best=L{best_layer+1}({best_auroc:.4f})"
                  f"  MI={mi:.4f}{flag}")
            del model; torch.cuda.empty_cache()

        # Aggregate over seeds with 1-based layer index
        layer_auroc_agg = {}
        for l in range(N_LAYERS):
            vals = all_layer_auroc[l]
            if vals:
                layer_auroc_agg[str(l+1)] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'std':  round(float(np.std(vals)),  4),
                }

        if delta_mids:
            valid = [s for s in per_seed.values() if s is not None]
            best_l = max(layer_auroc_agg,
                         key=lambda l: layer_auroc_agg[l]['mean']) \
                     if layer_auroc_agg else None
            direction_results[pe] = {
                'per_seed':      per_seed,
                'delta_entropy': {'mean': round(float(np.mean(delta_mids)),4),
                                  'std':  round(float(np.std(delta_mids)), 4)},
                'mid_auroc':     {'mean': round(float(np.mean([s['mid_auroc'] for s in valid])),4),
                                  'std':  round(float(np.std( [s['mid_auroc'] for s in valid])),4)},
                'mi_stability':  {'mean': round(float(np.mean(mis)),4),
                                  'std':  round(float(np.std(mis)), 4)},
                'layer_auroc':   layer_auroc_agg,
                'best_layer':    best_l,
            }

    return direction_results

# ── Main execution ────────────────────────────────────────────────────────────
print("Loading datasets...")
ds_in_a, ds_ood_a = get_datasets_a()
ds_in_b, ds_ood_b = get_datasets_b()
print(f"  Direction A — in:  ImageNet-100 val ({len(ds_in_a):,} images)")
print(f"  Direction A — ood: CIFAR-100 test   ({len(ds_ood_a):,} images)")
print(f"  Direction B — in:  CIFAR-100 test   ({len(ds_in_b):,} images)")
print(f"  Direction B — ood: ImageNet-100 val ({len(ds_ood_b):,} images)")

output = {}

output['smer_a'] = process_direction(
    "DIRECTION A: IN-100 models → C-100 images (Semantic Degradation)",
    RESULTS_IN, ds_in_a, ds_ood_a,
    num_workers_in=4, num_workers_ood=0
)

output['smer_b'] = process_direction(
    "DIRECTION B: C-100 models → IN-100 images (Structural Pressure)",
    RESULTS_CF, ds_in_b, ds_ood_b,
    num_workers_in=0, num_workers_ood=4
)

# ── Save JSON ─────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'cross_domain.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved: {out_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── TABLE 3: CROSS-DOMAIN RESULTS ──────────────────────────────────────")
for smer_key, smer_label in [('smer_a', 'Direction A: IN-100 → C-100'),
                              ('smer_b', 'Direction B: C-100 → IN-100')]:
    print(f"\n{smer_label}")
    print(f"{'PE':<14} {'Delta H':>12} {'Mid AUROC':>12} {'Best Layer':>12} {'MI':>10}")
    print("-" * 62)
    for pe in PE_TYPES:
        d = output[smer_key].get(pe)
        if not d: print(f"  {pe:<12} N/A"); continue
        de = d['delta_entropy']; da = d['mid_auroc']
        dm = d['mi_stability'];  bl = d['best_layer']
        bla = d['layer_auroc'].get(str(bl), {}).get('mean', float('nan')) if bl else float('nan')
        print(f"  {pe:<12} "
              f"{de['mean']:+.4f}±{de['std']:.4f}  "
              f"{da['mean']:.4f}±{da['std']:.4f}  "
              f"{bl}({bla:.3f})  "
              f"{dm['mean']:.4f}±{dm['std']:.4f}")

print("\n✓ Completed")

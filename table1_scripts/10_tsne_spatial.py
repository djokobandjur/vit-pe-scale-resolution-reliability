"""
SCRIPT 10 — Figure 3: t-SNE Spatial Awareness Panel (2x4)
===========================================================
Visualises the spatial organisation of last-block patch representations.
Bojenje po HSV gradijent patch poziciji — "smoking gun" za prostornu svest.

Panel:
  Redovi (2): IN-100 (gore), C-100 (dole)
  Kolone (4): Learned, Sinusoidal, RoPE, ALiBi

Output:
  results/figures/figure3_tsne_spatial.png  (300 DPI, za paper)
  results/figures/figure3_tsne_spatial.pdf  (za LaTeX)
  results/table1/tsne_embeddings.json       (metadata)
"""

import os, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVE_ROOT    = '/content/drive/My Drive/pe_experiment'
RESULTS_IN    = os.path.join(DRIVE_ROOT, 'results')
RESULTS_CF    = os.path.join(DRIVE_ROOT, 'results_cifar100')
DATA_IN       = '/content/imagenet100'
OUT_FIG_DIR   = os.path.join(DRIVE_ROOT, 'results', 'figures')
OUT_DATA_DIR  = os.path.join(DRIVE_ROOT, 'results', 'table1')
os.makedirs(OUT_FIG_DIR,  exist_ok=True)
os.makedirs(OUT_DATA_DIR, exist_ok=True)

PE_TYPES      = ['learned', 'sinusoidal', 'rope', 'alibi']
PE_LABELS     = {'learned': 'Learned', 'sinusoidal': 'Sinusoidal',
                 'rope': 'RoPE', 'alibi': 'ALiBi'}
SEEDS         = [42, 123, 456]
NUM_IMAGES    = 100    # IN-100
SEED_FOR_VIZ  = 42     # one seed for visualization
BATCH_SIZE    = 64
TSNE_PERP     = 40     
TSNE_ITER     = 1000
TSNE_SEED     = 42

IN_MEAN = [0.485, 0.456, 0.406]; IN_STD = [0.229, 0.224, 0.225]
CF_MEAN = [0.5071,0.4867,0.4408]; CF_STD = [0.2675,0.2565,0.2761]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Dataloaders ───────────────────────────────────────────────────────────────
def make_imagenet_loader():
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)
    ])
    ds     = datasets.ImageFolder(os.path.join(DATA_IN, 'val'), tf)
    # Fixed subset — consistent across all models (reproducibility)
    rng    = np.random.RandomState(TSNE_SEED)
    idx    = rng.choice(len(ds), size=min(NUM_IMAGES, len(ds)), replace=False)
    subset = Subset(ds, idx.tolist())
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)

def make_cifar_loader():
    tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CF_MEAN, CF_STD)
    ])
    ds     = datasets.CIFAR100('/content', train=False, download=True, transform=tf)
    rng    = np.random.RandomState(TSNE_SEED)
    idx    = rng.choice(len(ds), size=min(NUM_IMAGES, len(ds)), replace=False)
    subset = Subset(ds, idx.tolist())
    print(f"  CIFAR loader: {len(subset)} slika")  # ← dodati
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0,   # ← promeniti na 0 (bez workers)
                      pin_memory=False)

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

# ── Extract patch embeddings from the final block ─────────────────────────
@torch.no_grad()
def extract_last_block_patches(model, loader, n_patches):
    """
    Hook na model.blocks[-1] (poslednji blok).
    Vraca: np.array [N_img * n_patches, D] — samo patch tokeni, bez CLS.
    """
    storage = []

    def hook(module, input, output):
        acts = output[:, 1:, :]           # ukloni CLS [B, N, D]
        acts = acts[:, :n_patches, :]     # obreži
        storage.append(acts.cpu())

    h = model.blocks[-1].register_forward_hook(hook)

    for imgs, _ in tqdm(loader, desc="  extracting", leave=False):
        model(imgs.to(device))

    h.remove()

    if not storage:
        return None

    cat     = torch.cat(storage, dim=0)    # [B_total, N, D]
    B, N, D = cat.shape
    return cat.reshape(B * N, D).numpy()   # [B*N, D]

# ── HSV coloring by patch position ────────────────────────────────────────────────
def get_spatial_colors(n_patches_side):
    """
    Generiše HSV gradijent boje za n_patches_side x n_patches_side grid.
    Hue    = kolona (levo=plavo/0, desno=crveno/0.85)
    Value  = red    (gore=svetlo/1.0, dole=tamno/0.45)
    Saturacija = 0.9 (zivo, ne preterano)
    """
    N      = n_patches_side
    colors = np.zeros((N * N, 3))
    for r in range(N):
        for c in range(N):
            hue  = c / (N - 1) * 0.85      # 0.0 (plava) → 0.85 (ljubicasta)
            val  = 1.0 - (r / (N - 1)) * 0.55   # 1.0 (svetlo) → 0.45 (tamno)
            sat  = 0.9
            rgb  = mcolors.hsv_to_rgb([hue, sat, val])
            colors[r * N + c] = rgb
    return colors

# ── t-SNE projection ──────────────────────────────────────────────────────────
def run_tsne(feats, perplexity=TSNE_PERP, n_iter=TSNE_ITER, seed=TSNE_SEED):
    """
    feats: np.array [N, D]
    Vraca: np.array [N, 2]
    """
    # PCA before t-SNE for stability (reduced to 50 dims)
    from sklearn.decomposition import PCA
    n_comp = min(50, feats.shape[1], feats.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=seed)
    feats_pca = pca.fit_transform(feats)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init='pca',
        learning_rate='auto',
        random_state=seed,
        n_jobs=-1,
    )
    return tsne.fit_transform(feats_pca)

# ── Plotting ─────────────────────────────────────────────────────────────
def create_figure(all_embeddings, all_colors, n_patches_sides):
    """
    all_embeddings: {ds: {pe: np.array [N, 2]}}
    all_colors:     {ds: np.array [N, 3]}
    """
    datasets_list = ['imagenet100', 'cifar100']
    ds_labels     = {'imagenet100': 'ImageNet-100\n(224×224, patch=16)',
                     'cifar100':    'CIFAR-100\n(32×32, patch=4)'}

    fig = plt.figure(figsize=(22, 11))
    fig.patch.set_facecolor('white')

    gs = GridSpec(2, 4, figure=fig,
                  wspace=0.06, hspace=0.12,
                  left=0.07, right=0.88,
                  top=0.91, bottom=0.04)

    for r, ds in enumerate(datasets_list):
        n_side  = n_patches_sides[ds]
        colors  = all_colors[ds]      # [n_patches, 3] — boje za jedan set patcha

        for c, pe in enumerate(PE_TYPES):
            ax   = fig.add_subplot(gs[r, c])

            if ds not in all_embeddings or pe not in all_embeddings[ds]:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                continue

            emb    = all_embeddings[ds][pe]    # [N_img * n_patches, 2]
            n_pts  = emb.shape[0]
            n_p    = n_side ** 2

            # Repeat colors for each image
            n_imgs  = n_pts // n_p
            pt_cols = np.tile(colors, (n_imgs, 1))[:n_pts]

            # Scatter
            ax.scatter(emb[:, 0], emb[:, 1],
                       c=pt_cols, s=4, alpha=0.55,
                       linewidths=0, rasterized=True)

            # Styling
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('#cccccc')

            # Column titles
            if r == 0:
                ax.set_title(PE_LABELS[pe], fontsize=13, fontweight='bold',
                             pad=8, color='#1a1a1a')

            # Labele redova (levo)
            if c == 0:
                ax.set_ylabel(ds_labels[ds], fontsize=11, fontweight='bold',
                              labelpad=10, color='#1a1a1a')

    # ── CORRECT 2D LEGEND: Original Spatial Grid ──
    # Create a square on the right side of the figure
    legend_ax = fig.add_axes([0.90, 0.40, 0.08, 0.20]) # [x, y, width, height]
    
    # Generating a smooth 2D color map
    N_leg = 100
    leg_colors = np.zeros((N_leg, N_leg, 3))
    for r in range(N_leg):
        for c in range(N_leg):
            hue = c / (N_leg - 1) * 0.85          
            val = 1.0 - (r / (N_leg - 1)) * 0.55  
            leg_colors[r, c] = mcolors.hsv_to_rgb([hue, 0.9, val])
            
    legend_ax.imshow(leg_colors, aspect='auto', origin='upper')
    
    
    legend_ax.set_xticks([0, N_leg-1])
    legend_ax.set_xticklabels(['Left', 'Right'], fontsize=10, fontweight='bold')
    
    legend_ax.set_yticks([0, N_leg-1])
    legend_ax.set_yticklabels(['Top', 'Bottom'], fontsize=10, fontweight='bold')
    legend_ax.yaxis.tick_right() # Stavljamo labele za redove na desnu stranu
    
    legend_ax.set_title("Original Image Grid", fontsize=11, pad=12, fontweight='bold')
    legend_ax.set_xlabel("Hue changes with Column", fontsize=10, color='#333333')
    legend_ax.set_ylabel("Brightness changes with Row", fontsize=10, color='#333333', rotation=270, labelpad=15)
    legend_ax.yaxis.set_label_position("right")
    
    # Dodavanje tankog okvira oko legende
    for spine in legend_ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
        
        
    # ────────────────────────────────────────────────
    # Title
    fig.suptitle(
        'Spatial Awareness of Patch Representations (Last Block, t-SNE)\n'
        'Color = patch position (hue→column, brightness→row)',
        fontsize=12, y=0.97, color='#1a1a1a', style='italic'
    )

    return fig

# ── Main loop ─────────────────────────────────────────────────────────────
all_feats      = {}    # {ds: {pe: np.array [N*P, D]}}
all_embeddings = {}    # {ds: {pe: np.array [N*P, 2]}}
all_colors     = {}    # {ds: np.array [P, 3]}
n_patches_sides = {}

dataset_configs = [
    ('imagenet100', RESULTS_IN, make_imagenet_loader),
    ('cifar100',    RESULTS_CF, make_cifar_loader),
]

for ds_name, results_root, loader_fn in dataset_configs:
    print(f"\n{'='*60}\nDataset: {ds_name.upper()}\n{'='*60}")
    loader = loader_fn()
    all_feats[ds_name]      = {}
    all_embeddings[ds_name] = {}

    for pe in PE_TYPES:
        print(f"\n── {pe.upper()} ─────────────────────────────────────")
        model, n_patches = load_model(results_root, pe, SEED_FOR_VIZ)
        if model is None:
            continue

        n_side = int(n_patches ** 0.5)
        n_patches_sides[ds_name] = n_side

        # Extraction
        feats = extract_last_block_patches(model, loader, n_patches)
        del model; torch.cuda.empty_cache()

        if feats is None:
            print("  [FAILED] Nema aktivacija"); continue

        all_feats[ds_name][pe] = feats
        print(f"  Feats shape: {feats.shape}")

        # t-SNE
        print(f"  Running t-SNE (perplexity={TSNE_PERP}, n_iter={TSNE_ITER})...")
        emb = run_tsne(feats)
        all_embeddings[ds_name][pe] = emb
        print(f"  t-SNE done: {emb.shape}")

    # Boje za ovaj dataset
    n_side = n_patches_sides.get(ds_name, 14)
    all_colors[ds_name] = get_spatial_colors(n_side)

    del loader

# ── Creating and saving figure ───────────────────────────────────────────────
print("\nKreiram figure...")
fig = create_figure(all_embeddings, all_colors, n_patches_sides)

png_path = os.path.join(OUT_FIG_DIR, 'figure3_tsne_spatial.png')
pdf_path = os.path.join(OUT_FIG_DIR, 'figure3_tsne_spatial.pdf')

fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path,           bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"✓ PNG: {png_path}")
print(f"✓ PDF: {pdf_path}")

# Sacuvaj metadata
meta = {
    ds: {pe: {'shape': all_embeddings[ds][pe].shape[0],
               'n_patches': n_patches_sides.get(ds, 0) ** 2}
         for pe in PE_TYPES if pe in all_embeddings.get(ds, {})}
    for ds in ['imagenet100', 'cifar100']
}
meta_path = os.path.join(OUT_DATA_DIR, 'tsne_metadata.json')
with open(meta_path, 'w') as f:
    json.dump({'config': {'num_images': NUM_IMAGES, 'seed': SEED_FOR_VIZ,
                           'perplexity': TSNE_PERP, 'n_iter': TSNE_ITER,
                           'layer': 'last_block'}, 'datasets': meta}, f, indent=2)
print(f"✓ Metadata: {meta_path}")

print("\nCompleted! Figure3 saved in results/figures/")

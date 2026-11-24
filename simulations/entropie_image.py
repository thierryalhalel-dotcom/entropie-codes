"""L'entropie sous toutes ses formes -- chapitre « Entropie et arts visuels: la complexité du beau »."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import entropy as scipy_entropy
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def image_entropy(img_array):
    """Entropie de Shannon d'une image en niveaux de gris."""
    counts = np.bincount(img_array.flatten(), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return scipy_entropy(probs, base=2)

def local_entropy(img, window=5):
    """Carte d'entropie locale."""
    h, w = img.shape
    pad = window // 2
    img_pad = np.pad(img, pad, mode='reflect')
    H_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            patch = img_pad[i:i+window, j:j+window].flatten()
            counts = np.bincount(patch, minlength=256)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            H_map[i, j] = scipy_entropy(probs, base=2)
    return H_map

def box_counting_dim(img_bin, sizes=None):
    """Dimension fractale par box-counting sur image binaire."""
    if sizes is None:
        sizes = [2, 4, 8, 16, 32, 64]
    counts = []
    for s in sizes:
        # Réduction par blocs
        blocks = img_bin[:img_bin.shape[0]//s*s,
                         :img_bin.shape[1]//s*s]
        blocks = blocks.reshape(blocks.shape[0]//s, s,
                                blocks.shape[1]//s, s)
        counts.append((blocks.any(axis=(1, 3))).sum())
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# --- Génération d'exemples synthétiques ---
rng = np.random.default_rng(0)
N = 256

# Bruit blanc
noise = rng.integers(0, 256, (N, N), dtype=np.uint8)

# Image structurée (cercles concentriques)
x, y = np.meshgrid(np.arange(N), np.arange(N))
structured = ((np.sqrt((x-128)**2+(y-128)**2) % 20) * 12).clip(0,255).astype(np.uint8)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, img, title in zip(axes.flat,
    [noise, structured],
    ['Bruit blanc', 'Image structurée']):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    H = image_entropy(img)
    ax.set_title(f'{title}\n$H = {H:.2f}$ bits/pixel')
    ax.axis('off')

# Cartes d'entropie locale
for ax, img, title in zip(axes.flat[2:],
    [noise, structured],
    ['Entropie locale (bruit)', 'Entropie locale (structuré)']):
    H_map = local_entropy(img, window=7)
    im = ax.imshow(H_map, cmap='hot')
    plt.colorbar(im, ax=ax)
    ax.set_title(title); ax.axis('off')

plt.tight_layout()
plt.savefig(FIG / 'image_entropie.pdf')

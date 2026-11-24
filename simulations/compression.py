"""L'entropie sous toutes ses formes -- chapitre « Compression de données ».

Taux de compression mesures (zlib, lzma) sur des donnees de natures variees.
"""
import lzma
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG = RACINE / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(9)

texte = (RACINE / "donnees/trois_mousquetaires.txt").read_text(encoding="utf8")
texte = texte[texte.rindex("CHAPITRE PREMIER"):].encode("latin-1", "replace")[:1_000_000]
n = len(texte)
y, x = np.mgrid[0:1000, 0:1000] / 1000
image = (127 + 60 * np.sin(6 * x) * np.cos(4 * y) + rng.normal(0, 1.0, x.shape))
melange = bytes(rng.permutation(np.frombuffer(texte, np.uint8)))
donnees = {
    "motif\nrépété": (b"entropie " * n)[:n],
    "image\nlisse": image.astype(np.uint8).tobytes()[:n],
    "roman\nde Dumas": texte,
    "lettres du roman\nmélangées": melange,
    "bruit\nblanc": rng.integers(0, 256, n, dtype=np.uint8).tobytes(),
    "déjà\ncompressé": lzma.compress(texte),
}


def H0(b):
    p = np.array(list(Counter(b).values()), float) / len(b)
    return -(p * np.log2(p)).sum()


noms, h, z, l = list(donnees), [], [], []
for nom, b in donnees.items():
    h.append(H0(b))
    z.append(8 * len(zlib.compress(b, 9)) / len(b))
    l.append(8 * len(lzma.compress(b, preset=9)) / len(b))
    print(f"{nom!r:34} H0 = {h[-1]:.2f}  zlib = {z[-1]:.2f}  lzma = {l[-1]:.2f} bits/octet")
X = np.arange(len(noms))
plt.figure(figsize=(9.5, 4))
plt.bar(X - 0.27, h, 0.26, label="entropie des octets (ordre 0)", color="lightgray")
plt.bar(X, z, 0.26, label="zlib (ZIP)", color="steelblue")
plt.bar(X + 0.27, l, 0.26, label="lzma (7z)", color="firebrick")
plt.axhline(8, ls="--", color="gray", lw=1)
plt.xticks(X, noms, fontsize=9)
plt.ylabel("bits par octet")
plt.ylim(0, 10)
plt.legend(fontsize=8, ncol=3, loc="upper center")
plt.tight_layout()
plt.savefig(FIG / "compression.pdf")

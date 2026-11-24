"""L'entropie sous toutes ses formes -- chapitre « Percolation ».

Entropie de Shannon de la distribution des tailles d'amas, en fonction de p,
pour deux definitions : une voix par amas, ou une voix par site occupe.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(14)
PS = np.round(np.arange(0.30, 0.81, 0.02), 3)
PS = np.unique(np.append(PS, [0.57, 0.59, 0.593, 0.60, 0.61]))


def H(c):
    q = c[c > 0] / c.sum()
    return -(q * np.log2(q)).sum()


fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for L, n, coul in ((100, 60, "lightgray"), (200, 30, "steelblue"), (400, 12, "firebrick")):
    A, S = [], []
    for p in PS:
        a, s = [], []
        for _ in range(n):
            t = np.bincount(label(rng.random((L, L)) < p)[0].ravel())[1:]
            k = np.bincount(t)               # nombre d'amas de chaque taille
            a.append(H(k.astype(float)))
            s.append(H(k * np.arange(len(k), dtype=float)))
        A.append(np.mean(a))
        S.append(np.mean(s))
    print(L, "max par amas a p =", PS[np.argmax(A)], round(max(A), 2),
          "| par site a p =", PS[np.argmax(S)], round(max(S), 2))
    ax[0].plot(PS, A, "o-", ms=3, color=coul, label=f"$L$ = {L}")
    ax[1].plot(PS, S, "o-", ms=3, color=coul, label=f"$L$ = {L}")
for a, t in zip(ax, ("une voix par amas", "une voix par site occupé")):
    a.axvline(0.593, ls="--", color="gray", lw=1)
    a.set_xlabel("$p$")
    a.set_ylabel("$H$ des tailles d'amas (bits)")
    a.set_title(t)
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "percolation_entropie.pdf")

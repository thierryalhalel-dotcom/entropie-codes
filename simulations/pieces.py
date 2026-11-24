"""L'entropie sous toutes ses formes -- chapitre « Boltzmann : le désordre se compte ».

Boite de pieces secouee : combien de pile ? Comparaison avec le denombrement.
"""
from math import comb, log
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(1)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for N, couleur in ((20, "steelblue"), (100, "firebrick")):
    n = rng.binomial(N, 0.5, 200)                    # 200 secousses
    f = np.bincount(n, minlength=N + 1) / 200
    ax[0].bar(np.arange(N + 1) / N, f, width=0.8 / N, color=couleur, alpha=0.5,
              label=f"{N} pièces (200 secousses)")
    ax[0].plot(np.arange(N + 1) / N, [comb(N, k) / 2**N for k in range(N + 1)],
               "-", color=couleur, lw=1.2)
    print(N, "moyenne", n.mean(), "ecart-type", n.std().round(2),
          "theorie", round(np.sqrt(N) / 2, 2))
ax[0].set_xlabel("Proportion de piles $n/N$")
ax[0].set_ylabel("Fréquence")
ax[0].legend(fontsize=8)

# on part de 100 faces et on retourne 5 pieces prises au hasard a chaque fois
N = 100
etat = np.zeros(N, dtype=int)
S, piles = [], []
for _ in range(120):
    piles.append(etat.sum())
    S.append(log(comb(N, int(etat.sum()))))
    k = rng.choice(N, 5, replace=False)
    etat[k] ^= 1
ax[1].plot(S, color="forestgreen")
ax[1].axhline(log(comb(N, N // 2)), ls="--", color="gray", label="maximum ($n = 50$)")
ax[1].set_xlabel("Nombre de tours (5 pièces retournées par tour)")
ax[1].set_ylabel(r"$S/k_B = \ln\,\Omega(n)$")
ax[1].legend()
plt.tight_layout()
plt.savefig(FIG / "pieces.pdf")

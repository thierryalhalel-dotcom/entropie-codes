"""L'entropie sous toutes ses formes -- chapitre « Entropie et évolution darwinienne »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(10)
N, GEN, REP = 50, 300, 400               # N diploides : 2N copies du gene
pops = rng.multinomial(2 * N, [0.25] * 4, REP)
pops[:] = [2 * N // 4] * 4
X = np.empty((GEN + 1, REP, 4))
X[0] = pops / (2 * N)
for g in range(GEN):
    X[g + 1] = [rng.multinomial(2 * N, p) / (2 * N) for p in X[g]]
q = np.where(X > 0, X, 1)
H = -(X * np.log2(q)).sum(2)
het = 1 - (X**2).sum(2)
t = np.arange(GEN + 1)
fixe = (X.max(2) == 1).mean(1)
print("fixees a 100, 300 :", fixe[100], fixe[300], "H moyen 0,100 :", H[0].mean(), H[100].mean().round(2))
print("het 100 :", het[100].mean().round(3), "theorie", round(0.75 * (1 - 1 / (2 * N))**100, 3))
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
for k, c in zip(range(4), ("firebrick", "steelblue", "forestgreen", "orange")):
    ax[0].plot(t, X[:, 0, k], color=c, lw=1.2)
ax[0].set_ylabel("Fréquences des 4 allèles")
ax[0].set_title("une population")
ax[1].plot(t, H[:, :12], color="gray", lw=0.5)
ax[1].plot(t, H.mean(1), color="firebrick", lw=2, label="moyenne de 400")
ax[1].set_ylabel("$H$ (bits)")
ax[1].legend()
ax[2].plot(t, het.mean(1), color="steelblue", lw=2, label="simulation")
ax[2].plot(t, 0.75 * (1 - 1 / (2 * N))**t, "k--", lw=1, label=r"$0{,}75\,(1-1/2N)^t$")
ax[2].set_ylabel("Hétérozygotie")
ax[2].legend()
for a in ax:
    a.set_xlabel("Génération")
plt.tight_layout()
plt.savefig(FIG / "derive_genetique.pdf")

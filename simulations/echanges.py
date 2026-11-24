"""L'entropie sous toutes ses formes -- chapitre « Entropie en économie et sciences sociales »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(11)
N, M0, TOURS = 5000, 100, 4000
m = np.full(N, M0)
H, G, temps = [], [], []


def gini(v):
    v = np.sort(v)
    n = len(v)
    return 2 * (np.arange(1, n + 1) * v).sum() / (n * v.sum()) - (n + 1) / n


def entropie(v):
    p = np.bincount(v // 10) / N         # tranches de 10 euros
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


garde = {}
for t in range(TOURS + 1):
    if t % 20 == 0:
        temps.append(t)
        H.append(entropie(m))
        G.append(gini(m))
    if t in (50, TOURS):
        garde[t] = m.copy()
    perm = rng.permutation(N)
    a, b = perm[:N // 2], perm[N // 2:]  # a donne 5 euros a b, s'il les a
    ok = m[a] >= 5
    m[a[ok]] -= 5
    m[b[ok]] += 5
print("Gini final", round(G[-1], 3), "H finale", round(H[-1], 2),
      "H expo", round(np.log2(np.e * M0 / 10), 2), "part sous la moyenne", (m < M0).mean())
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
for t, c in ((50, "lightgray"), (TOURS, "steelblue")):
    ax[0].hist(garde[t], bins=np.arange(0, 700, 10), density=True, color=c, alpha=0.8,
               label=f"après {t} tours")
x = np.linspace(0, 700, 300)
ax[0].plot(x, np.exp(-x / M0) / M0, "r-", lw=1.5, label=r"$e^{-x/\mu}/\mu$")
ax[0].set_ylim(0, 0.018)
ax[0].set_xlabel("Avoir (euros)")
ax[0].set_ylabel("Densité")
ax[0].legend(fontsize=8)
ax[1].plot(temps, H, color="firebrick")
ax[1].set_ylabel("$H$ (bits, tranches de 10 euros)")
ax[2].plot(temps, G, color="forestgreen")
ax[2].axhline(0.5, ls="--", color="gray", lw=1)
ax[2].set_ylabel("Coefficient de Gini")
for a in ax[1:]:
    a.set_xlabel("Tours")
plt.tight_layout()
plt.savefig(FIG / "echanges.pdf")

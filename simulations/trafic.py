"""L'entropie sous toutes ses formes -- chapitre « Trafic routier ».

Automate de Nagel-Schreckenberg sur un anneau : embouteillage fantome,
diagramme fondamental et entropie des vitesses.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(5)
L, VMAX, P = 400, 5, 0.25


def simule(rho, n, chauffe=600):
    N = int(round(rho * L))
    x = np.sort(rng.choice(L, N, replace=False))
    v = np.zeros(N, int)
    X, V = [], []
    for t in range(chauffe + n):
        ecart = (np.roll(x, -1) - x - 1) % L
        v = np.minimum(np.minimum(v + 1, VMAX), ecart)
        v = np.maximum(v - (rng.random(N) < P), 0)
        x = (x + v) % L
        if t >= chauffe:
            X.append(x.copy())
            V.append(v.copy())
    return np.array(X), np.array(V)


fig, ax = plt.subplots(1, 3, figsize=(13, 4))
X, V = simule(0.18, 250)
t = np.repeat(np.arange(250), X.shape[1])
sc = ax[0].scatter(X.ravel(), t, c=V.ravel(), s=1, cmap="RdYlGn", vmin=0, vmax=VMAX)
plt.colorbar(sc, ax=ax[0], label="vitesse (cases par pas)")
ax[0].set_xlabel("Position (cases)")
ax[0].set_ylabel("Temps (pas)")
ax[0].set_title(r"$\rho$ = 0,18 : bouchons fantômes")
rhos = np.linspace(0.03, 0.9, 40)
q, H = [], []
for r in rhos:
    _, V = simule(r, 300)
    q.append(V.mean() * V.shape[1] / L)
    p = np.bincount(V.ravel(), minlength=VMAX + 1) / V.size
    p = p[p > 0]
    H.append(-(p * np.log2(p)).sum())
ax[1].plot(rhos, q, "o-", ms=3, color="steelblue")
ax[1].set_xlabel(r"Densité $\rho$ (voitures par case)")
ax[1].set_ylabel("Débit $q$ (voitures par pas)")
ax[2].plot(rhos, H, "o-", ms=3, color="firebrick")
ax[2].set_xlabel(r"Densité $\rho$")
ax[2].set_ylabel("$H_v$ (bits)")
ax[2].set_title("Entropie des vitesses")
for a in ax[1:]:
    a.axvline(rhos[int(np.argmax(q))], ls="--", color="gray", lw=1)
plt.tight_layout()
plt.savefig(FIG / "trafic.pdf", dpi=200)
print("rho_c", rhos[int(np.argmax(q))].round(3), "q_max", max(q).__round__(3),
      "H max", max(H).__round__(2), "a rho", rhos[int(np.argmax(H))].round(2))
X, _ = simule(0.18, 250)

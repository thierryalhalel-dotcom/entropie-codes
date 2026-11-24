"""L'entropie sous toutes ses formes -- chapitre « Le vivant ».

Une « proteine » de 14 residus (modele HP, reseau carre), traitee EXACTEMENT :
on enumere toutes les conformations, puis on calcule U, S, F et C.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
SEQ = "HPHPPHHPHPPHPH"
N = len(SEQ)
PAS = ((1, 0), (0, 1), (-1, 0), (0, -1))
niveaux, meilleurs = Counter(), {}


def explore(chemin, occ, E, plan):
    """Toutes les marches auto-evitantes, aux rotations et reflexions pres."""
    k = len(chemin)
    if k == N:
        niveaux[E] += 1
        meilleurs.setdefault(E, list(chemin))
        return
    x, y = chemin[-1]
    for dx, dy in PAS:
        if not plan and dy < 0:
            continue                         # reflexion
        q = (x + dx, y + dy)
        if q in occ:
            continue
        e = E
        if SEQ[k] == "H":
            for ax, ay in PAS:
                j = occ.get((q[0] + ax, q[1] + ay))
                if j is not None and j < k - 1 and SEQ[j] == "H":
                    e -= 1
        occ[q] = k
        chemin.append(q)
        explore(chemin, occ, e, plan or dy != 0)
        chemin.pop()
        del occ[q]


explore([(0, 0), (1, 0)], {(0, 0): 0, (1, 0): 1}, 0, False)
E = np.array(sorted(niveaux), float)
g = np.array([niveaux[int(e)] for e in E], float)
print(dict(sorted(niveaux.items())), "total", int(g.sum()), "S max", np.log(g.sum()).round(2))
T = np.linspace(0.08, 1.5, 400)
w = g[None] * np.exp(-(E - E[0])[None] / T[:, None])
Z = w.sum(1)
p = w / Z[:, None]
U = (p * E).sum(1)
C = ((p * E**2).sum(1) - U**2) / T**2
S = np.log(Z) + (U - E[0]) / T
pn = p[:, 0]
i = C.argmax()
print("pic de C a kT =", T[i].round(2), "C =", C[i].round(2), "p natif =", pn[i].round(2),
      "p=1/2 a kT =", T[np.argmin(abs(pn - 0.5))].round(2))
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
c = np.array(meilleurs[int(E[0])])
ax[0].plot(*c.T, "k-", lw=1.5, zorder=1)
for (a, b), s in zip(c, SEQ):
    ax[0].scatter(a, b, s=110, c="firebrick" if s == "H" else "white", edgecolor="k", zorder=2)
ax[0].set_aspect("equal")
ax[0].axis("off")
ax[0].set_title(f"un état natif ($E$ = {int(E[0])}, {int(g[0])} au total)", fontsize=10)
ax[1].semilogy(E, g, "o-", color="steelblue")
ax[1].set_xlabel("Énergie $E$ (en contacts H-H)")
ax[1].set_ylabel("Nombre de conformations $\\Omega(E)$")
ax[2].plot(T, pn, color="firebrick", label="$p$ (état natif)")
ax[2].plot(T, S / np.log(g.sum()), color="steelblue", label=r"$S/S_{\max}$")
ax[2].set_xlabel("$k_BT$")
ax[2].legend(fontsize=8)
ax[3].plot(T, C, "k-")
ax[3].set_xlabel("$k_BT$")
ax[3].set_ylabel("$C/k_B$")
plt.tight_layout()
plt.savefig(FIG / "proteine_exacte.pdf")

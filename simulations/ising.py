"""L'entropie sous toutes ses formes -- chapitre « Transitions de phase ».

Modele d'Ising 2D (Metropolis en damier), compare a la solution d'Onsager.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(4)
Tc = 2 / np.log(1 + np.sqrt(2))


def balayages(s, T, n):
    L = len(s)
    damier = np.add.outer(np.arange(L), np.arange(L)) % 2
    for _ in range(n):
        for c in (0, 1):
            h = (np.roll(s, 1, 0) + np.roll(s, -1, 0)
                 + np.roll(s, 1, 1) + np.roll(s, -1, 1))
            ok = rng.random(s.shape) < np.exp(np.minimum(0, -2 * s * h / T))
            s = np.where(ok & (damier == c), -s, s)
    return s


fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
for a, T, nom in zip(ax, (1.8, Tc, 3.5), ("$T < T_c$", "$T = T_c$", "$T > T_c$")):
    s = balayages(np.ones((128, 128), int) if T < 2 else rng.choice([-1, 1], (128, 128)),
                  T, 1500)
    a.imshow(s, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    a.set_title(nom + f" ({T:.2f})".replace(".", ","))
    a.set_xticks([])
    a.set_yticks([])
Ts = np.linspace(1.2, 3.6, 25)
m = []
s = np.ones((48, 48), int)
for T in Ts:
    s = balayages(s, T, 400)
    v = []
    for _ in range(150):
        s = balayages(s, T, 4)
        v.append(abs(s.mean()))
    m.append(np.mean(v))
x = np.linspace(1.2, Tc - 1e-6, 400)
ax[3].plot(x, (1 - np.sinh(2 / x)**-4)**0.125, "k-", lw=1, label="exact (Onsager-Yang)")
ax[3].plot([Tc, 3.6], [0, 0], "k-", lw=1)
ax[3].plot(Ts, m, "o", ms=4, color="firebrick", label=r"simulation $48\times48$")
ax[3].set_xlabel("$k_BT/J$")
ax[3].set_ylabel("$|m|$")
ax[3].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "ising.pdf", dpi=200)
print("|m| a 2,1 et 2,4 :", np.interp([2.1, 2.4], Ts, m).round(2))

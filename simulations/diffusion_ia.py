"""L'entropie sous toutes ses formes -- chapitre « L'IA générative ».

Modele de diffusion dans le plan, avec un score EXACT (aucun reseau) :
les donnees sont un melange de 8 gaussiennes disposees en cercle.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(6)
N, T, s0 = 1500, 400, 0.08
ang = 2 * np.pi * np.arange(8) / 8
mu = 1.3 * np.c_[np.cos(ang), np.sin(ang)]
beta = np.linspace(1e-4, 0.03, T)
abar = np.cumprod(1 - beta)


def score(x, t):
    """Gradient de ln p_t pour le melange bruite (forme analytique)."""
    v = abar[t] * s0**2 + 1 - abar[t]
    d = np.sqrt(abar[t]) * mu[None] - x[:, None]
    w = -(d**2).sum(-1) / (2 * v)
    w = np.exp(w - w.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * d).sum(1) / v


def entropie(x, n=24, a=3.2):
    h = np.histogram2d(*x.T, bins=n, range=[[-a, a]] * 2)[0].ravel()
    p = h[h > 0] / h.sum()
    return -(p * np.log2(p)).sum()


x0 = mu[rng.integers(0, 8, N)] + s0 * rng.normal(size=(N, 2))
vues = [0, 60, 150, T]
fig, ax = plt.subplots(2, 4, figsize=(12, 6.4), sharex=True, sharey=True)
for a, t in zip(ax[0], vues):
    x = x0 if t == 0 else (np.sqrt(abar[t - 1]) * x0
                           + np.sqrt(1 - abar[t - 1]) * rng.normal(size=x0.shape))
    a.plot(*x.T, ".", ms=2, color="steelblue")
    a.set_title(f"$t$ = {t}, $H$ = {entropie(x):.1f} bits".replace(".", ","))
    print("aller", t, round(entropie(x), 2))
x = rng.normal(size=(N, 2))
etats = {T: x.copy()}
for t in range(T - 1, -1, -1):
    x = (x + beta[t] * score(x, t)) / np.sqrt(1 - beta[t])
    if t > 0:
        x += np.sqrt(beta[t]) * rng.normal(size=x.shape)
    if t in vues:
        etats[t] = x.copy()
for a, t in zip(ax[1], vues[::-1]):
    a.plot(*etats[t].T, ".", ms=2, color="firebrick")
    a.set_title(f"$t$ = {t}, $H$ = {entropie(etats[t]):.1f} bits".replace(".", ","))
    print("retour", t, round(entropie(etats[t]), 2))
for a in ax.ravel():
    a.set_xlim(-3.2, 3.2)
    a.set_ylim(-3.2, 3.2)
    a.set_aspect("equal")
ax[0, 0].set_ylabel("bruitage (aller)")
ax[1, 0].set_ylabel("débruitage (retour)")
plt.tight_layout()
plt.savefig(FIG / "diffusion_ia.pdf")

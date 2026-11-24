"""L'entropie sous toutes ses formes -- chapitre « Entropie en médecine »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(13)
N = 1000


def colore(beta):
    f = np.fft.rfftfreq(N)
    s = np.fft.rfft(rng.normal(size=N))
    s[1:] *= f[1:]**(-beta / 2)
    s[0] = 0
    x = np.fft.irfft(s, N)
    return x / x.std()


def sampen(x, m=2, r=0.2):
    r = r * x.std()
    E = np.array([x[i:N - m + i] for i in range(m + 1)]).T
    def paires(k):
        d = np.abs(E[:, None, :k] - E[None, :, :k]).max(2)
        return ((d < r).sum() - len(E)) / 2
    return -np.log(paires(m + 1) / paires(m))


n = np.arange(N)
series = {"quasi périodique": 800 + 40 * (np.sin(2 * np.pi * n / 20) * 1.35 + 0.15 * rng.normal(size=N)),
          "fluctuations en $1/f$": 800 + 40 * colore(1.0),
          "totalement aléatoire": 800 + 40 * rng.normal(size=N)}
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
for a, (nom, x), c in zip(ax, series.items(), ("firebrick", "forestgreen", "steelblue")):
    v = sampen(x)
    print(nom, "SampEn", round(v, 2), "moyenne", x.mean().round(0), "ecart-type", x.std().round(0))
    a.plot(x[:200], color=c, lw=0.9)
    a.set_title(f"{nom}\nSampEn = {v:.2f}".replace(".", ","), fontsize=10)
    a.set_xlabel("Battement n°")
ax[0].set_ylabel("Intervalle RR (ms)")
plt.tight_layout()
plt.savefig(FIG / "sampen.pdf")

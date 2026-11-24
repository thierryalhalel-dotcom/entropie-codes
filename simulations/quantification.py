"""L'entropie sous toutes ses formes -- chapitre « Entropie différentielle »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
t = np.linspace(0, 1, 200_000, endpoint=False)
x = np.sin(2 * np.pi * 3.1234 * t)


def quantifie(x, B):
    d = 2 / 2**B
    return np.clip(np.floor(x / d) * d + d / 2, -1 + d / 2, 1 - d / 2)


B = np.arange(1, 17)
snr, H = [], []
for b in B:
    y = quantifie(x, b)
    snr.append(10 * np.log10(np.mean(x**2) / np.mean((x - y)**2)))
    n = np.unique(y, return_counts=True)[1] / len(y)
    H.append(-(n * np.log2(n)).sum())
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
q = t < 0.33
ax[0].plot(t[q], x[q], "k-", lw=0.8)
for b, c in ((2, "firebrick"), (4, "steelblue")):
    ax[0].plot(t[q], quantifie(x, b)[q], color=c, lw=1.2, label=f"{b} bits")
ax[0].set_xlabel("Temps")
ax[0].legend()
ax[1].plot(B, snr, "o", color="firebrick", label="mesuré")
ax[1].plot(B, 6.02 * B + 1.76, "k-", lw=1, label="6,02 $B$ + 1,76")
ax[1].set_xlabel("Nombre de bits $B$")
ax[1].set_ylabel("SNR (dB)")
ax[1].legend()
h = np.log2(np.pi / 2)                   # entropie differentielle d'un sinus (bits)
ax[2].plot(B[:12], H[:12], "o", color="steelblue", label="mesurée")
ax[2].plot(B[:12], h + B[:12] - 1, "k-", lw=1, label=r"$h(X) + B - \log_2(2A)$")
ax[2].set_xlabel("Nombre de bits $B$")
ax[2].set_ylabel("$H$ du signal quantifié (bits)")
ax[2].legend()
print("SNR 8 bits", round(snr[7], 1), "16 bits", round(snr[15], 1), "H 8 bits", round(H[7], 2), "pred", round(h + 7, 2))
plt.tight_layout()
plt.savefig(FIG / "quantification.pdf")

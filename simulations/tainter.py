"""L'entropie sous toutes ses formes -- chapitre « Entropie des civilisations »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
C = np.linspace(0, 5, 300)
B = 1 - np.exp(-C)                       # benefice, B0 = C0 = 1
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(C, B, "k-", lw=2, label="bénéfice $B(C)$")
for k, c in ((0.1, "steelblue"), (0.3, "firebrick")):
    copt = np.log(1 / k)
    ax[0].plot(C, k * C, color=c, ls="--", label=f"coût $kC$, $k$ = {k}".replace(".", ","))
    ax[0].plot([copt] * 2, [k * copt, 1 - k], color=c, lw=3, alpha=0.6)
    ax[1].plot(C, B - k * C, color=c, label=f"$k$ = {k}".replace(".", ","))
    ax[1].plot(copt, 1 - k - k * copt, "o", color=c)
    print(k, "C* =", round(copt, 2), "net =", round(1 - k - k * copt, 2),
          "net avec k=0,3 a l'ancien optimum :", round(1 - 0.1 - 0.3 * np.log(10), 2))
c1 = np.log(10)
ax[1].annotate("", xy=(c1, 1 - 0.1 - 0.3 * c1), xytext=(c1, 1 - 0.1 - 0.1 * c1),
               arrowprops=dict(arrowstyle="->", lw=1.5))
ax[1].text(c1 + 0.1, 0.4, "choc :\n$k$ triple", fontsize=9)
ax[1].axhline(0, color="gray", lw=0.8)
for a in ax:
    a.set_xlabel("Complexité $C/C_0$")
    a.legend(fontsize=8)
ax[0].set_ylabel("Bénéfice et coût (unité $B_0$)")
ax[1].set_ylabel("Bénéfice net $B - kC$")
plt.tight_layout()
plt.savefig(FIG / "tainter.pdf")

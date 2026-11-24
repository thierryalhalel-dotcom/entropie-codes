"""L'entropie sous toutes ses formes -- chapitre « Ensembles statistiques et fonction de partition »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
x = np.linspace(0.02, 5, 600)            # kT / epsilon
p1 = 1 / (np.exp(1 / x) + 1)
S = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
C = p1 * (1 - p1) / x**2
i = C.argmax()
print(f"C max = {C[i]:.3f} k_B a kT = {x[i]:.3f} epsilon ; S(5) = {S[-1]:.3f}")
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
for a, y, l in zip(ax, (p1, S, C), ("$p_1$ (niveau excité)", "$S/k_B$", "$C/k_B$")):
    a.plot(x, y, color="firebrick", lw=2)
    a.set_xlabel(r"$k_BT/\varepsilon$")
    a.set_ylabel(l)
ax[0].axhline(0.5, ls="--", color="gray", lw=1)
ax[1].axhline(np.log(2), ls="--", color="gray", lw=1, label=r"$\ln 2$")
ax[1].legend()
plt.tight_layout()
plt.savefig(FIG / "deux_niveaux.pdf")

"""L'entropie sous toutes ses formes -- chapitre « Le principe holographique »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
lP = 1.616e-35
R = np.logspace(-15, 27, 400)
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.loglog(R, np.pi * (R / lP)**2, "k-", lw=2, label=r"borne holographique $A/4\ell_P^2$ (aire)")
n = 1e29                                 # particules par m3 (matiere condensee)
ax.loglog(R, n * 4 / 3 * np.pi * R**3, color="steelblue", lw=1.5,
          label=r"matière ordinaire, $\sim 1\,k_B$ par particule (volume)")
Rx = 3 * np.pi / lP**2 / (4 * np.pi * n)
pts = {"Terre": (6.4e6, 1e50), "Soleil": (7e8, 1e58),
       "trou noir solaire": (3e3, 1e77), "Sgr A*": (1.2e10, 1.7e90),
       "Univers observable": (4.4e26, 3e104)}
for nom, (r, s) in pts.items():
    ax.plot(r, s, "o", color="firebrick")
    ax.annotate(nom, (r, s), xytext=(6, -11), textcoords="offset points", fontsize=8)
ax.set_xlabel("Rayon (m)")
ax.set_ylabel("$S/k_B$")
ax.set_ylim(1, 1e130)
ax.set_xlim(1e-15, 1e33)
ax.legend()
print(f"croisement R = {Rx:.1e} m")
plt.tight_layout()
plt.savefig(FIG / "loi_aire.pdf")

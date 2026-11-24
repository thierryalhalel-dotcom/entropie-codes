"""L'entropie sous toutes ses formes -- chapitre « Potentiels thermodynamiques »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
dH, dS = -92e3, -199.0          # J/mol, J/(mol K), synthese de l'ammoniac
T = np.linspace(250, 800, 300)
G = dH - T * dS
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(T, np.full_like(T, dH / 1e3), color="firebrick", label=r"$\Delta_rH^\circ$")
ax[0].plot(T, -T * dS / 1e3, color="steelblue", label=r"$-T\,\Delta_rS^\circ$")
ax[0].plot(T, G / 1e3, "k-", lw=2, label=r"$\Delta_rG^\circ$")
ax[0].axhline(0, color="gray", lw=0.8)
ax[0].axvline(dH / dS, ls="--", color="gray", lw=1)
ax[0].set_xlabel("Température (K)")
ax[0].set_ylabel("kJ/mol")
ax[0].legend()
ax[1].semilogy(T, np.exp(-G / (8.314 * T)), "k-", lw=2)
ax[1].axhline(1, color="gray", lw=0.8)
ax[1].axvline(dH / dS, ls="--", color="gray", lw=1, label="$K = 1$")
ax[1].set_xlabel("Température (K)")
ax[1].set_ylabel("Constante d'équilibre $K$")
ax[1].legend()
for t in (298, 462, 700):
    print(t, round((dH - t * dS) / 1e3, 1), f"{np.exp(-(dH - t*dS)/(8.314*t)):.2e}")
plt.tight_layout()
plt.savefig(FIG / "haber.pdf")

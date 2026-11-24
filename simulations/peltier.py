"""L'entropie sous toutes ses formes -- chapitre « Onsager ».

Analyse des deux experiences sur un module Peltier (donnees SIMULEES) :
generateur (Seebeck) puis pompe a chaleur (inversion du courant).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG, DON = RACINE / "figures", RACINE / "donnees"
FIG.mkdir(exist_ok=True)
DON.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
alpha, Rm, Km = 0.050, 2.0, 0.50          # V/K, ohm, W/K (valeurs typiques)
Cb, I, T0 = 180.0, 1.0, 293.15            # bloc d'aluminium de 200 g (J/K)


def sonde(x):
    return np.round((x + rng.normal(0, 0.03, np.shape(x))) / 0.0625) * 0.0625


# 1. Seebeck : bloc chaud qui refroidit, radiateur a 20 degres, circuit ouvert
dT = np.linspace(30, 2, 60)
dTm = sonde(dT)
Vm = np.round((alpha * dT + rng.normal(0, 0.002, 60)) / 0.004) * 0.004  # pas 4 mV
a1, b1 = np.polyfit(dTm, Vm, 1)
np.savetxt(DON / "peltier_seebeck_SIMULE.csv", np.c_[dTm, Vm], delimiter=",",
           header="dT_K,V", comments="", fmt="%.4f")


# 2. Peltier : bloc a 20 degres, courant +I puis -I, radiateur ventile a 20 degres
def marche(signe, n=41, dt=1.0):
    T = [T0]
    for _ in range(n - 1):
        Q = -signe * alpha * I * T[-1] + Rm * I**2 / 2 - Km * (T[-1] - T0)
        T.append(T[-1] + dt * Q / Cb)
    return dt * np.arange(n), sonde(np.array(T) - 273.15)


t, froid = marche(+1)
_, chaud = marche(-1)
np.savetxt(DON / "peltier_pompe_SIMULE.csv", np.c_[t, froid, chaud], delimiter=",",
           header="t_s,T_plus_C,T_moins_C", comments="", fmt="%.4f")
# demi-difference : l'effet Joule (pair en I) s'elimine ; pente initiale (parabole)
D = (chaud - froid) / 2
p = np.polyfit(t, D, 2)
pente = p[1]
a2 = Cb * pente / (I * T0)
moy = np.polyfit(t, (chaud + froid) / 2, 2)[1] * Cb
tir = []
for _ in range(300):
    _, f = marche(+1)
    _, c = marche(-1)
    tir.append(Cb * np.polyfit(t, (c - f) / 2, 2)[1] / (I * T0))
print(f"alpha Seebeck = {1e3*a1:.1f} mV/K ; alpha Peltier = {1e3*a2:.1f} mV/K "
      f"(u = {1e3*np.std(tir):.1f}) ; pente {pente:.4f} K/s ; P = {Cb*pente:.1f} W ; "
      f"Joule {moy:.2f} W (attendu {Rm*I**2/2})")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(dTm, Vm, ".", color="steelblue")
ax[0].plot(dTm, a1 * dTm + b1, "k-", lw=1,
           label=rf"$\alpha$ = {1e3*a1:.1f} mV/K".replace(".", ","))
ax[0].set_xlabel(r"$\Delta T$ (K)")
ax[0].set_ylabel("Tension à vide (V)")
ax[0].set_title("Seebeck : la chaleur crée une tension")
ax[0].legend()
ax[1].plot(t, froid, ".", color="steelblue", label="$+I$ : le bloc refroidit")
ax[1].plot(t, chaud, ".", color="firebrick", label="$-I$ : le bloc chauffe")
ax[1].plot(t, (froid + chaud) / 2, color="gray", lw=1, label="moyenne : effet Joule")
ax[1].set_xlabel("Temps (s)")
ax[1].set_ylabel("Température du bloc (°C)")
ax[1].set_title("Peltier : le courant transporte de l'entropie")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "peltier.pdf")

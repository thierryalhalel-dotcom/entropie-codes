"""L'entropie sous toutes ses formes -- chapitre « La flèche du temps ».

Analyse du refroidissement d'un cafe mesure avec la sonde de l'entropimetre.
Donnees SIMULEES : tasse avec et sans couvercle (evaporation), sonde DS18B20.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG, DON = RACINE / "figures", RACINE / "donnees"
FIG.mkdir(exist_ok=True)
DON.mkdir(exist_ok=True)
rng = np.random.default_rng(2)
K0, Ta, T0, C = 273.15, 20.0, 85.0, 0.200 * 4186 + 80   # cafe + tasse (J/K)
dt, n = 10.0, 541                                        # 90 min


def p_sat(T):
    """Pression de vapeur saturante (Pa), formule de Magnus."""
    return 611.2 * np.exp(17.62 * T / (243.12 + T))


def simule(G, k_evap):
    T = [T0]
    for _ in range(n - 1):
        perte = G * (T[-1] - Ta) + k_evap * (p_sat(T[-1]) - 0.5 * p_sat(Ta))
        T.append(T[-1] - dt * perte / C)
    T = np.array(T)
    return T, np.round((T + rng.normal(0, 0.03, n)) / 0.0625) * 0.0625


t = dt * np.arange(n)
cas = {"avec couvercle": (0.35, 0.0, "firebrick"),
       "sans couvercle": (0.35, 6e-4, "steelblue")}
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4))
for nom, (G, k, coul) in cas.items():
    _, Tm = simule(G, k)
    np.savetxt(DON / f"cafe_{nom.replace(' ', '_')}_SIMULE.csv", np.c_[t, Tm],
               delimiter=",", header="t_s,T_C", comments="", fmt="%.4f")
    ax[0].plot(t / 60, Tm, color=coul, label=nom)
    q = Tm - Ta > 3
    y = np.log(Tm[q] - Ta)
    ax[1].plot(t[q] / 60, y, color=coul, lw=1)
    debut, fin = t[q] < 600, t[q] > t[q][-1] - 1200
    tau1 = -1 / np.polyfit(t[q][debut], y[debut], 1)[0] / 60
    tau2 = -1 / np.polyfit(t[q][fin], y[fin], 1)[0] / 60
    dT = np.diff(Tm)
    Tmoy = (Tm[1:] + Tm[:-1]) / 2 + K0
    Sc = np.r_[0, np.cumsum(C * dT / Tmoy)]
    Se = np.r_[0, np.cumsum(-C * dT / (Ta + K0))]
    ax[2].plot(t / 60, Sc + Se, color=coul, label=nom)
    print(f"{nom}: tau debut {tau1:.1f} min, fin {tau2:.1f} min ; T(90) = {Tm[-1]:.1f} ; "
          f"dS_cafe = {Sc[-1]:.0f}, dS_env = {Se[-1]:.0f}, dS_uni = {Sc[-1]+Se[-1]:.1f} J/K")
lim = C * ((T0 - Ta) / (Ta + K0) - np.log((T0 + K0) / (Ta + K0)))
print("limite", round(lim, 1))
ax[2].axhline(lim, ls="--", color="gray", lw=1)
for a, l in zip(ax, ("Température (°C)", r"$\ln(T - T_a)$", r"$\Delta S_{uni}$ (J/K)")):
    a.set_xlabel("Temps (min)")
    a.set_ylabel(l)
ax[0].legend()
plt.tight_layout()
plt.savefig(FIG / "cafe_esp32.pdf")

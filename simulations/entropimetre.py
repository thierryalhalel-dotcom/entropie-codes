"""L'entropie sous toutes ses formes -- chapitre « L'entropimètre ».

1. SIMULE les mesures (chauffe, refroidissement, melange) : bruit, resolution
   de la sonde, pertes thermiques.
2. ANALYSE ces mesures comme de vraies mesures : aucune valeur tabulee
   (ni c, ni les masses) n'est utilisee.
Les CSV sont ecrits dans donnees/ (donnees SIMULEES).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG = RACINE / "figures"
DON = RACINE / "donnees"
FIG.mkdir(exist_ok=True)
DON.mkdir(exist_ok=True)
rng = np.random.default_rng(3)
K0 = 273.15

# ------------------------------------------------------------------
# 1. Simulation
# ------------------------------------------------------------------
c, m, mu = 4186.0, 0.200, 0.020      # J/(kg K), kg, kg (vase : equivalent eau)
G, Ta = 0.12, 20.0                   # W/K, degres C
U, I = 12.0, 2.5                     # V, A
C1 = c * (m + mu)                    # eau froide + vase
dt = 2.0


def sonde(T):
    """DS18B20 : bruit puis arrondi au pas de 0,0625 degre."""
    T = np.asarray(T) + rng.normal(0, 0.03, np.shape(T))
    return np.round(T / 0.0625) * 0.0625


def evolue(T0, C, puissance, n):
    T = [T0]
    for _ in range(n):
        T.append(T[-1] + dt * (puissance(T[-1]) - G * (T[-1] - Ta)) / C)
    return np.array(T)


# chauffe de 20 a 80 degres, puis refroidissement libre
T_vrai = [20.0]
while T_vrai[-1] < 80.0:
    T_vrai.append(T_vrai[-1] + dt * (U*I - G*(T_vrai[-1] - Ta)) / C1)
n_ch = len(T_vrai)
T_vrai = np.concatenate([T_vrai, evolue(T_vrai[-1], C1, lambda T: 0.0, 600)[1:]])
t = dt * np.arange(len(T_vrai))
chauffe = t < t[n_ch - 1]
Tm = sonde(T_vrai)
Um = U + rng.normal(0, 0.01, len(t))
Im = I + rng.normal(0, 0.005, len(t))
np.savetxt(DON / "entropimetre_chauffe_SIMULE.csv",
           np.c_[t, Tm, Um, Im, chauffe], delimiter=",",
           header="t_s,T_C,U_V,I_A,chauffe", comments="", fmt="%.4f")

# melange : 200 g a 80 degres verses dans (200 g + vase) a 20 degres
T1v, T2v = 20.0, 80.0
C2 = c * m
Tfv = (C1*T1v + C2*T2v) / (C1 + C2)
avant = np.full(60, T1v)
apres = evolue(Tfv, C1 + C2, lambda T: 0.0, 299)
tau = 8.0                            # retard de la sonde (s)
tp = dt * np.arange(300)
apres_lu = apres - (Tfv - T1v) * np.exp(-tp / tau)
tmel = dt * np.arange(360) - 120.0
Tmel = sonde(np.concatenate([avant, apres_lu]))
T2m = float(sonde(np.full(20, T2v)).mean())
np.savetxt(DON / "entropimetre_melange_SIMULE.csv", np.c_[tmel, Tmel],
           delimiter=",", header="t_s,T_C", comments="", fmt="%.4f")

# ------------------------------------------------------------------
# 2. Analyse
# ------------------------------------------------------------------
P = np.where(chauffe, Um * Im, 0.0)
Ta_m = Tm[:5].mean()                 # l'eau part de la temperature ambiante

# moyenne glissante pour les derivees
def lisse(x, k=15):
    y = np.pad(x, k // 2, mode="edge")
    return np.convolve(y, np.ones(k) / k, mode="valid")

# pertes : refroidissement libre -> G/C ; chauffe -> C
libre = ~chauffe & (t > t[n_ch - 1] + 20)
a = -np.polyfit(t[libre], np.log(Tm[libre] - Ta_m), 1)[0]     # G/C
Tl = lisse(Tm)
ch = np.where(chauffe)[0]
En = np.sum(P[ch]) * dt
aire = np.sum(Tl[ch] - Ta_m) * dt
C1m = En / ((Tl[ch[-1]] - Tl[ch[0]]) + a * aire)
Gm = a * C1m
print(f"G = {Gm:.3f} W/K (vrai {G}), C1 = {C1m:.0f} J/K (vrai {C1:.0f})")

# entropie : integrale de (P - G (T - Ta)) dt / T
brut = np.cumsum(P * dt / (Tl + K0))
corr = np.cumsum((P - Gm * (Tl - Ta_m)) * dt / (Tl + K0))
corr -= corr[0]
theo = C1 * np.log((T_vrai + K0) / (T_vrai[0] + K0))
print(f"S fin de chauffe : brut {brut[ch[-1]]:.1f}, corrige {corr[ch[-1]]:.1f},"
      f" theorie {theo[ch[-1]]:.1f} J/K ; duree {t[ch[-1]]:.0f} s")


def S_cal(T):
    """Courbe d'etalonnage S(T), lue par interpolation (T en degres C)."""
    return np.interp(T, Tl[ch], corr[ch])


# melange
T1m = Tmel[:60].mean()
q = tmel > 60
pente, Tfm = np.polyfit(tmel[q], Tmel[q], 1)   # extrapolation a t = 0


def cree(T1, T2, Tf):
    r = (Tf - T1) / (T2 - Tf)        # C2/C1, par conservation de l'energie
    return (S_cal(Tf) - S_cal(T1)) + r * (S_cal(Tf) - S_cal(T2)), r


dS, r = cree(T1m, T2m, Tfm)
dS1 = S_cal(Tfm) - S_cal(T1m)
dS2 = r * (S_cal(Tfm) - S_cal(T2m))
vrai = C1*np.log((Tfv+K0)/(T1v+K0)) + C2*np.log((Tfv+K0)/(T2v+K0))
print(f"T1={T1m:.2f} T2={T2m:.2f} Tf={Tfm:.2f} r={r:.3f} (vrai {C2/C1:.3f})")
print(f"froide {dS1:+.1f}  chaude {dS2:+.1f}  creee {dS:+.2f} J/K (vrai {vrai:.2f})")

# incertitudes : Monte-Carlo sur les trois temperatures
u = 0.1
tir = np.array([cree(T1m + rng.normal(0, u), T2m + rng.normal(0, u),
                     Tfm + rng.normal(0, u))[0] for _ in range(20000)])
print(f"u = {tir.std():.2f} J/K ; decalage commun de 0,5 degre : "
      f"{cree(T1m + .5, T2m - .0, Tfm + .5)[0] - dS:+.2f} si T2 exclue")
Cm = C1m


def cree_an(T1, T2, Tf):
    r = (Tf - T1) / (T2 - Tf)
    return Cm * (np.log((Tf+K0)/(T1+K0)) + r*np.log((Tf+K0)/(T2+K0)))


print(f"decalage commun 0,5 : {cree_an(T1m+.5, T2m+.5, Tfm+.5)-cree_an(T1m, T2m, Tfm):+.3f}")
print(f"Tf seule +0,5 : {cree_an(T1m, T2m, Tfm+.5)-cree_an(T1m, T2m, Tfm):+.3f}")

# ------------------------------------------------------------------
# 3. Figures
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(t / 60, Tm, color="firebrick", lw=1)
ax[0].axvline(t[ch[-1]] / 60, color="gray", ls="--", lw=1)
ax[0].text(t[ch[-1]] / 60 - 1, 25, "chauffe", ha="right")
ax[0].text(t[ch[-1]] / 60 + 1, 25, "refroidissement")
ax[0].set_xlabel("Temps (min)")
ax[0].set_ylabel("Température (°C)")
ax[1].plot(Tl[ch], brut[ch], color="gray", ls="--", label="sans correction des pertes")
ax[1].plot(Tl[ch], corr[ch], color="steelblue", lw=2, label="corrigée des pertes")
ax[1].plot(T_vrai[ch][::80], theo[ch][::80], "k.", ms=6, label=r"$C\,\ln(T/T_0)$")
ax[1].set_xlabel("Température (°C)")
ax[1].set_ylabel("$S - S(20$ °C$)$ (J/K)")
ax[1].legend()
plt.tight_layout()
plt.savefig(FIG / "entropimetre_etalonnage.pdf")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(tmel, Tmel, ".", ms=3, color="steelblue")
x = np.array([0, 600])
ax[0].plot(x, Tfm + pente * x, "r-", lw=1.2, label="extrapolation")
ax[0].plot(0, Tfm, "ro")
ax[0].annotate(f"$T_f$ = {Tfm:.2f} °C".replace(".", ","), (0, Tfm), xytext=(110, Tfm - 9),
               arrowprops=dict(arrowstyle="->"))
ax[0].set_xlabel("Temps (s)")
ax[0].set_ylabel("Température (°C)")
ax[0].legend(loc="lower right")
ax[1].plot(Tl[ch], corr[ch], color="steelblue", lw=2, label="$S$ mesurée (1 vase)")
ax[1].plot([T1m, T2m], S_cal([T1m, T2m]), "k--", lw=1, label="corde")
w = r / (1 + r)
corde = (1 - w) * S_cal(T1m) + w * S_cal(T2m)
ax[1].plot([Tfm, Tfm], [corde, S_cal(Tfm)], color="firebrick", lw=3)
ax[1].annotate("écart à la corde\n" + rf"$\Delta S/(1+r)$",
               (Tfm, (corde + S_cal(Tfm)) / 2), xytext=(54, 45),
               arrowprops=dict(arrowstyle="->"))
ax[1].plot([T1m, T2m, Tfm], S_cal([T1m, T2m, Tfm]), "ko")
for T, n, d in ((T1m, "$T_1$", (0, 12)), (T2m, "$T_2$", (-18, 0)), (Tfm, "$T_f$", (-18, 5))):
    ax[1].annotate(n, (T, S_cal(T)), textcoords="offset points", xytext=d)
ax[1].set_xlabel("Température (°C)")
ax[1].set_ylabel("$S - S(20$ °C$)$ (J/K)")
ax[1].legend(loc="upper left")
plt.tight_layout()
plt.savefig(FIG / "entropimetre_melange.pdf")

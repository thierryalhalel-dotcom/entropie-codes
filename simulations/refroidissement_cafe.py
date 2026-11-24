"""L'entropie sous toutes ses formes -- chapitre « La flèche du temps »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


# Parametres physiques
T_amb = 20.0      # degres Celsius
T_0   = 85.0      # degres Celsius
tau   = 15.0      # constante de temps (minutes)
m     = 0.200     # kg (masse du cafe)
c_eau = 4186.0    # J / (kg.K)

t  = np.linspace(0, 90, 900)    # 90 minutes
T  = T_amb + (T_0 - T_amb) * np.exp(-t / tau)    # en degres C

# Conversion en kelvin pour les calculs d'entropie
T_K     = T + 273.15
T_amb_K = T_amb + 273.15

# Variation d'entropie cumulee (integration numerique par trapeze)
dt = t[1] - t[0]

# dS_cafe = m*c * dT / T  --> integrate
dT = np.gradient(T_K, t)            # dT/dt
dS_cafe_dt = m * c_eau * dT / T_K  # W/K par minute
S_cafe = np.cumsum(dS_cafe_dt) * dt

# dS_env = -m*c * dT / T_amb  (l'env recoit -dQ)
dS_env_dt = -m * c_eau * dT / T_amb_K
S_env = np.cumsum(dS_env_dt) * dt

S_uni = S_cafe + S_env

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(t, T, color='firebrick', lw=2, label='Café')
ax1.axhline(T_amb, ls='--', color='steelblue', lw=1.5,
            label=f'Ambiance ({T_amb:.0f} °C)')
ax1.set_ylabel('Température (°C)')
ax1.legend()
ax1.set_title(
    f'Refroidissement : $T_0 = {T_0:.0f}$ °C, '
    f'$T_{{\\rm amb}} = {T_amb:.0f}$ °C, $\\tau = {tau:.0f}$ min')

ax2.plot(t, S_uni,  color='forestgreen', lw=2,
         label=r'$\Delta S_{\rm uni}$ (total)')
ax2.plot(t, S_cafe, color='firebrick',   lw=1.5, ls='--',
         label=r'$\Delta S_{\rm café}$')
ax2.plot(t, S_env,  color='steelblue',  lw=1.5, ls=':',
         label=r'$\Delta S_{\rm env}$')
ax2.axhline(0, color='k', lw=0.8, alpha=0.5)
ax2.set_ylabel('Entropie (J/K)')
ax2.set_xlabel('Temps (min)')
ax2.legend()

# Calcul analytique du DS_total final
T_f = T_amb_K
DS_cafe_tot = m * c_eau * np.log(T_f / (T_0 + 273.15))
DS_env_tot  = m * c_eau * (T_0 - T_amb) / T_amb_K
DS_uni_tot  = DS_cafe_tot + DS_env_tot
ax2.annotate(
    f'$\\Delta S_{{\\rm uni}}^{{\\rm final}} = {DS_uni_tot:.3f}$ J/K',
    xy=(85, DS_uni_tot), xytext=(55, DS_uni_tot * 0.7),
    arrowprops=dict(arrowstyle='->', color='forestgreen'),
    fontsize=9, color='forestgreen')

plt.tight_layout()
plt.savefig(FIG / 'fleche_temps_cafe.pdf', dpi=150)

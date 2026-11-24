"""L'entropie sous toutes ses formes -- chapitre « Gastronomie: thermodynamique dans l'assiette »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


# --- Courbe de denaturation de l'albumine ---
# DeltaG(T) = DeltaH - T * DeltaS
DH = 450e3      # J/mol
DS = 1350       # J/mol/K
T = np.linspace(273, 373, 200)      # 0 a 100 degres C
DG = DH - T * DS                    # energie libre (J/mol)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.plot(T - 273, DG / 1e3, color='steelblue', lw=2,
        label=r'$\Delta G = \Delta H - T\,\Delta S$')
T_denat = DH / DS
ax.axvline(T_denat - 273, color='firebrick', lw=1.5, ls='--',
           label=f'$T_{{dén}} = {T_denat - 273:.0f}$°C')
ax.fill_between(T - 273, DG / 1e3, 0,
                where=(DG < 0), alpha=0.15, color='firebrick',
                label='Dénaturation spontanée ($\\Delta G < 0$)')
ax.set_xlabel('Température (°C)')
ax.set_ylabel('$\\Delta G$ (kJ/mol)')
ax.set_title('Dénaturation de l\'albumine\n'
             r'$\Delta H = 450$ kJ/mol, $\Delta S = 1350$ J/mol/K')
ax.legend(fontsize=9)

# --- Profil aromatique et entropie ---
ax = axes[1]
familles = ['Fruité', 'Épicé', 'Boisé', 'Floral', 'Minéral', 'Animal']
vin_complex = np.array([0.35, 0.20, 0.15, 0.15, 0.10, 0.05])
vin_simple  = np.array([0.70, 0.20, 0.10, 0,    0,    0   ])

def H_shan(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

H_c = H_shan(vin_complex)
H_s = H_shan(vin_simple)

x = np.arange(len(familles))
w = 0.35
ax.bar(x - w/2, vin_complex, w, label=f'Vin complexe  H={H_c:.2f} bits',
       color='steelblue', alpha=0.8)
ax.bar(x + w/2, vin_simple,  w, label=f'Vin simple    H={H_s:.2f} bit',
       color='salmon', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(familles, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Fraction aromatique (sans dim.)')
ax.set_title('Entropie aromatique de deux vins\n'
             r'$H = -\sum p_i \log_2 p_i$')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG / 'gastronomie_entropie.pdf', dpi=150)

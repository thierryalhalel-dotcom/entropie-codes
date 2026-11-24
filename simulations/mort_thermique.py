"""L'entropie sous toutes ses formes -- chapitre « La mort thermique : le destin entropique de l'Univers »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


log_t = np.array([
    -43,    # temps de Planck (secondes)
    -36,    # inflation
    2,      # nucleosynthese
    12,     # recombinaison (380 000 ans ~ 10^{12} s)
    17.6,   # aujourd'hui (13.8 Ga ~ 4.35e17 s)
])

log_S = np.array([
    88,     # entropie post-Planck (estimee)
    89,     # apres inflation (photons + particules)
    89,     # nucleosynthese (peu de changement)
    90,     # recombinaison
    104,    # aujourd'hui (trous noirs supermassifs)
])

log_t_futur = np.array([
    17.6,   # aujourd'hui
    21.5,   # 10^{14} ans : fin des etoiles
    47.5,   # 10^{40} ans : desintegration proton
    74.5,   # 10^{67} ans : evaporation trous noirs stellaires
    107.5,  # 10^{100} ans : evaporation derniers trous noirs
    157.5,  # 10^{150} ans : ere sombre
])

log_S_futur = np.array([
    104,    # aujourd'hui
    105,    # fin des etoiles
    108,    # desintegration proton
    115,    # evaporation trous noirs stellaires
    121,    # evaporation derniers trous noirs
    122,    # equilibre de De Sitter (asymptote)
])

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(log_t, log_S, 'o-', color='steelblue', lw=2,
        label='Passé', markersize=6)
ax.plot(log_t_futur, log_S_futur, 's--', color='firebrick', lw=2,
        label='Futur', markersize=6)

ax.axhline(122, ls=':', color='gray', lw=1)
ax.text(60, 122.5, r'$S_{\mathrm{DS}} \approx 10^{122}\,k_B$',
        color='gray', fontsize=10)

labels = ['Planck', 'Inflation', r'Nucl.', 'CMB',
          "Aujourd'hui"]
for i, lab in enumerate(labels):
    ax.annotate(lab, (log_t[i], log_S[i]),
                textcoords='offset points', xytext=(5, 10),
                fontsize=8)

labels_f = ['', r'Fin des étoiles', 'Désint. proton', 'TN stellaires',
            'Derniers TN', r'De Sitter']
for i, lab in enumerate(labels_f):
    if lab:
        ax.annotate(lab, (log_t_futur[i], log_S_futur[i]),
                    textcoords='offset points', xytext=(5, -15),
                    fontsize=8, color='firebrick')

ax.set_xlabel(r'$\log_{10}(t / \mathrm{s})$')
ax.set_ylabel(r'$\log_{10}(S / k_B)$')
ax.set_title("Chronologie entropique de l'Univers")
ax.legend()
ax.set_xlim(-48, 175)
ax.set_ylim(85, 125)

plt.tight_layout()
plt.savefig(FIG / 'mort_thermique.pdf', dpi=150)

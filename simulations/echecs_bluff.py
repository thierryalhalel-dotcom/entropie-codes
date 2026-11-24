"""L'entropie sous toutes ses formes -- chapitre « Entropie et jeux: les échecs, le poker, le go »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# --- Entropie d'une position en fonction de la phase de jeu ---
# Phases : ouverture, milieu, fin de partie
phases = np.linspace(0, 1, 200)    # 0 = ouverture, 1 = finale

# Nombre de coups legaux approximatif (parabole avec pic en milieu de partie)
n_coups = 5 + 30 * 4 * phases * (1 - phases)

# Distribution simulee : quelques coups bons + queue longue
# H_pos approximee par H d'une loi geometrique sur n_coups symboles
# H_geo = -(p log p + (1-p) log(1-p)) * n avec p decroissant
# On utilise une approximation simple : H ~ log2(n_eff)
n_eff = 1 + (n_coups - 1) * 0.35   # "nombre de coups vraiment jouables"
H_pos = np.log2(np.maximum(n_eff, 1))

ax = axes[0]
ax.plot(phases, n_coups, color='steelblue', lw=2,
        label='Coups légaux $n$')
ax_r = ax.twinx()
ax_r.plot(phases, H_pos, color='firebrick', lw=2, ls='--',
          label='$H_{pos}$ (bits)')
ax.set_xlabel('Phase de jeu (0 = ouverture, 1 = finale)')
ax.set_ylabel('Nombre de coups légaux', color='steelblue')
ax_r.set_ylabel('$H_{pos}$ (bits)', color='firebrick')
ax.set_title('Entropie d\'une position aux échecs\n'
             r'$H_{pos} = -\sum_i p_i \log_2 p_i$')
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc='upper right')

# --- Bluff optimal : entropie en fonction du taux de bluff ---
p_bluff = np.linspace(1e-6, 1 - 1e-6, 200)
H_bluff = np.where(
    (p_bluff > 1e-9) & (p_bluff < 1 - 1e-9),
    -(p_bluff * np.log2(p_bluff) + (1 - p_bluff) * np.log2(1 - p_bluff)),
    0)

# Plusieurs configurations (b, c)
configs = [(10, 10, 'b=10, c=10'), (10, 30, 'b=10, c=30'), (10, 90, 'b=10, c=90')]

ax = axes[1]
ax.plot(p_bluff, H_bluff, color='gray', lw=1.5, ls='--',
        label='Entropie max (p=0,5)')
for b, c, label in configs:
    p_opt = b / (b + c)
    H_opt = -(p_opt * np.log2(p_opt) + (1 - p_opt) * np.log2(1 - p_opt))
    ax.axvline(p_opt, color=None, lw=0)
    ax.plot(p_opt, H_opt, 'o', ms=9, label=f'{label}  $p^*$={p_opt:.2f}')

ax.set_xlabel('Fréquence de bluff $p$ (sans dim.)')
ax.set_ylabel('$H^* = -p\\log_2 p - (1-p)\\log_2(1-p)$ (bits)')
ax.set_title('Entropie de la stratégie de bluff\nà l\'équilibre de Nash')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG / 'echecs_entropie.pdf', dpi=150)

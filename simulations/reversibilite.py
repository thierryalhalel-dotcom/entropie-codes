"""L'entropie sous toutes ses formes -- chapitre « Le démon de Maxwell »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


N, dt, n_pas = 50, 0.05, 100
np.random.seed(7)
# Toutes les particules dans le quart inférieur gauche
pos = np.random.rand(N, 2) * 0.5          # positions (sans dim.)
vel = np.random.randn(N, 2) * 0.1         # vitesses (u.a./pas)

def evoluer(pos, vel, n_pas, dt):
    """Évolution avec conditions aux limites périodiques."""
    for _ in range(n_pas):
        pos = (pos + vel * dt) % 1.0      # conditions périodiques
    return pos

pos_final = evoluer(pos.copy(), vel.copy(), n_pas, dt)
# Retournement des vitesses : inversion du temps
pos_retour = evoluer(pos_final.copy(), -vel.copy(), n_pas, dt)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for ax, p, titre in zip(axes,
    [pos, pos_final, pos_retour],
    ['État initial\n(localisé)',
     f'Après {n_pas} pas\n(dispersé)',
     'Retour (t inversé)\n(relocalisé)']):
    ax.scatter(p[:,0], p[:,1], s=15, color='navy', alpha=0.7)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(titre, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (sans dim.)')
    ax.set_ylabel('$y$ (sans dim.)')

plt.suptitle(
    'Réversibilité microscopique --- $N=50$ particules\n'
    'Retourner les vitesses ramène à l\'état initial',
    fontsize=10)
plt.tight_layout()
plt.savefig(FIG / 'reversibilite.pdf', dpi=150)

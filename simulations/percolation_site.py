"""L'entropie sous toutes ses formes -- chapitre « Percolation: l'ordre qui émerge du hasard »."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def percole(N, p):
    """True si la grille NxN de probabilité p percole."""
    grille = np.random.rand(N, N) < p   # cases ouvertes
    etiq, _ = label(grille)             # composantes connexes
    haut = set(etiq[0,  :]) - {0}       # amas touchant le haut
    bas  = set(etiq[-1, :]) - {0}       # amas touchant le bas
    return bool(haut & bas)             # chemin haut-bas ?

N, n_essais = 100, 40
# p sans dimension, entre 0 et 1
p_vals = np.linspace(0.4, 0.8, 60)
prob = [np.mean([percole(N, p) for _ in range(n_essais)])
        for p in p_vals]

plt.figure(figsize=(6, 4))
plt.plot(p_vals, prob, 'o-', color='steelblue', ms=4, lw=1.5)
plt.axvline(0.593, ls='--', color='firebrick', lw=1.5,
            label=r'$p_c \approx 0{,}593$')
plt.xlabel('Probabilité $p$ (sans dimension)')
plt.ylabel('Probabilité de percolation (sans dimension)')
plt.legend()
plt.tight_layout()
plt.savefig(FIG / 'percolation_transition.pdf', dpi=150)

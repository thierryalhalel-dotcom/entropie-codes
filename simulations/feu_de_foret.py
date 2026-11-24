"""L'entropie sous toutes ses formes -- chapitre « Percolation: l'ordre qui émerge du hasard »."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def feu_foret(N, p, graine=None):
    """
    Propagation BFS depuis le bord gauche.
    Retourne : 0=vide, 1=arbre sain, 2=brûlé.
    """
    rng = np.random.default_rng(graine)
    foret = (rng.random((N, N)) < p).astype(int)  # 1=arbre, 0=vide

    file = deque()
    for i in range(N):
        if foret[i, 0] == 1:
            foret[i, 0] = 2          # allumage bord gauche
            file.append((i, 0))

    while file:
        i, j = file.popleft()
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0<=ni<N and 0<=nj<N and foret[ni,nj]==1:
                foret[ni, nj] = 2
                file.append((ni, nj))
    return foret

cmap_feu = ListedColormap(['white', 'forestgreen', 'firebrick'])
configs = [(0.45, '$p=0{,}45<p_c$'),
           (0.593,'$p=p_c\\approx0{,}59$'),
           (0.70, '$p=0{,}70>p_c$')]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (p, titre) in zip(axes, configs):
    carte = feu_foret(200, p, graine=0)
    ax.imshow(carte, cmap=cmap_feu, vmin=0, vmax=2,
              interpolation='nearest')
    pct = (carte==2).sum() / 200**2
    ax.set_title(f'{titre}\n{pct:.0%} brûlé', fontsize=10)
    ax.axis('off')

fig.suptitle('Feux de forêt --- grille $200\\times200$\n'
             '$p$ = fraction d\'arbres (sans dimension)')
plt.tight_layout()
plt.savefig(FIG / 'feu_foret.pdf', dpi=150)

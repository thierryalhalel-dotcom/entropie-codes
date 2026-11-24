"""L'entropie sous toutes ses formes -- chapitre « Percolation: l'ordre qui émerge du hasard »."""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def percole_lien(N, p):
    """Percolation de lien par BFS depuis le bord haut."""
    h = np.random.rand(N, N - 1) < p   # liaisons horizontales
    v = np.random.rand(N - 1, N) < p   # liaisons verticales

    visite = np.zeros((N, N), dtype=bool)
    file = deque([(0, j) for j in range(N)])
    for noeud in file:
        visite[noeud] = True

    while file:
        i, j = file.popleft()
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if not (0<=ni<N and 0<=nj<N and not visite[ni,nj]):
                continue
            # Vérification du lien selon la direction
            if di == -1 and v[ni, j]:   lien = True
            elif di ==  1 and v[i,  j]: lien = True
            elif dj == -1 and h[i, nj]: lien = True
            elif dj ==  1 and h[i,  j]: lien = True
            else:                        lien = False
            if lien:
                visite[ni, nj] = True
                file.append((ni, nj))
    return visite[-1, :].any()

N, n_essais = 80, 30
p_vals = np.linspace(0.3, 0.7, 50)
prob = [np.mean([percole_lien(N, p) for _ in range(n_essais)])
        for p in p_vals]

plt.plot(p_vals, prob, 's-', color='sienna', ms=4, lw=1.5)
plt.axvline(0.5, ls='--', color='firebrick', lw=1.5,
            label='$p_c = 0{,}5$ (exact, sans dim.)')
plt.xlabel('Fraction de liaisons ouvertes $p$ (sans dim.)')
plt.ylabel('Probabilité de percolation (sans dim.)')
plt.legend()
plt.savefig(FIG / 'percolation_lien.pdf', dpi=150)

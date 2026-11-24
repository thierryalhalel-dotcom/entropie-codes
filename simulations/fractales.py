"""L'entropie sous toutes ses formes -- chapitre « Entropie et fractales »."""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


# -------------------------------------------------------
# Partie 1 : flocon de Koch par generation recursive
# -------------------------------------------------------

def subdivise(pts):
    """
    Un passage Koch : remplace chaque segment par 4 sous-segments.
    pts : liste de points complexes.
    """
    nouveaux = []
    for A, B in zip(pts[:-1], pts[1:]):
        P = A + (B - A) / 3
        Q = A + (B - A) * 2 / 3
        M = P + (Q - P) * np.exp(-1j * np.pi / 3)  # sommet vers l'exterieur
        nouveaux += [A, P, M, Q]
    nouveaux.append(pts[-1])
    return nouveaux

# Triangle equilateral initial
a, b, c = 0+0j, 1+0j, 0.5 + np.sqrt(3)/2*1j
pts = [a, b, c, a]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for ax, n in zip(axes, [0, 1, 2, 4]):
    tmp = pts.copy()
    for _ in range(n):
        tmp = subdivise(tmp)
    xs = [p.real for p in tmp]
    ys = [p.imag for p in tmp]
    ax.plot(xs, ys, 'k-', lw=0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Étape {n}  ($L/L_0={(4/3)**n:.2f}$)')
plt.tight_layout()
plt.savefig(FIG / 'fractales_koch.pdf', dpi=150)

# -------------------------------------------------------
# Partie 2 : box-counting sur le flocon
# -------------------------------------------------------

def points_koch(n_iter=6, n_pts=50000):
    """Points uniformement distribues sur le flocon (etape n_iter)."""
    tmp = [a, b, c, a]
    for _ in range(n_iter):
        tmp = subdivise(tmp)
    # interpolation uniforme le long du trace
    segs = list(zip(tmp[:-1], tmp[1:]))
    longueurs = np.array([abs(B - A) for A, B in segs])
    longueurs /= longueurs.sum()
    idx = np.random.choice(len(segs), size=n_pts, p=longueurs)
    t = np.random.rand(n_pts)
    pts_arr = np.array(
        [(segs[i][0] + t[k]*(segs[i][1]-segs[i][0]))
         for k, i in enumerate(idx)]
    )
    return pts_arr.real, pts_arr.imag

def box_counting(xs, ys, eps_vals):
    """N(eps) pour chaque eps dans eps_vals."""
    N_vals = []
    for eps in eps_vals:
        bx = np.floor(xs / eps).astype(int)
        by = np.floor(ys / eps).astype(int)
        N_vals.append(len(set(zip(bx, by))))
    return np.array(N_vals)

xs, ys = points_koch()
eps_vals = np.logspace(-0.2, -1.8, 20)
N_vals = box_counting(xs, ys, eps_vals)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(np.log(1/eps_vals), np.log(N_vals), s=25, color='steelblue')
# ajustement lineaire
coef = np.polyfit(np.log(1/eps_vals), np.log(N_vals), 1)
ax.plot(np.log(1/eps_vals),
        np.polyval(coef, np.log(1/eps_vals)),
        'r--', lw=1.5, label=f'$D \\approx {coef[0]:.3f}$')
ax.set_xlabel(r'$\ln(1/\varepsilon)$')
ax.set_ylabel(r'$\ln N(\varepsilon)$')
ax.set_title('Box-counting sur le flocon de Koch')
ax.legend()
plt.tight_layout()
plt.savefig(FIG / 'fractales_boxcount.pdf', dpi=150)

# -------------------------------------------------------
# Partie 3 : agrégat DLA
# -------------------------------------------------------

def dla(N_part=3000, taille=200):
    """Agregat DLA par marche aleatoire."""
    rng = np.random.default_rng(42)
    grille = np.zeros((taille, taille), dtype=bool)
    cx, cy = taille // 2, taille // 2
    grille[cx, cy] = True          # graine centrale

    for _ in range(N_part):
        # lancer depuis un cercle de rayon r_lancer
        r = np.max(np.argwhere(grille).max(axis=0)
                   - np.array([cx, cy])) + 10
        r = min(r, taille // 2 - 5)
        theta = rng.uniform(0, 2*np.pi)
        x = int(cx + r * np.cos(theta))
        y = int(cy + r * np.sin(theta))

        while True:
            step = rng.integers(0, 4)
            dx, dy = [(1,0),(-1,0),(0,1),(0,-1)][step]
            x += dx;  y += dy
            # trop loin : particule perdue
            if abs(x-cx) >= taille//2-1 or abs(y-cy) >= taille//2-1:
                break
            # colle si voisin occupe
            voisins = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
            if any(0<=nx<taille and 0<=ny<taille and grille[nx,ny]
                   for nx, ny in voisins):
                grille[x, y] = True
                break

    return grille

grille_dla = dla()
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(~grille_dla, cmap='gray', interpolation='nearest')
ax.axis('off')
ax.set_title('Agrégat DLA (~3000 particules, $D\\approx1{,}71$)')
plt.tight_layout()
plt.savefig(FIG / 'fractales_dla.pdf', dpi=150)

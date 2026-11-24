#!/usr/bin/env python3
"""
generer_figures_nouveaux.py
Génère les figures des chapitres chap04c_ergodicite et chap07e_fractales.
Sortie : dossier figures/
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import deque
import os

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

def save(nom):
    plt.savefig(os.path.join(OUT, nom + ".pdf"), bbox_inches='tight')
    plt.close()
    print(f"  -> {nom}.pdf")


# ============================================================
# CHAPITRE ERGODICITE
# ============================================================

# ------------------------------------------------------------
# ergo_billard : billard circulaire vs billard de Bunimovich
# ------------------------------------------------------------
print("ergo_billard...")

def reflexion_cercle(pos, vel):
    """Réflexion spéculaire sur le cercle unité."""
    n = pos / np.linalg.norm(pos)       # normale sortante
    return vel - 2 * np.dot(vel, n) * n

def prochain_cercle(pos, vel, R=1.0):
    """Intersection d'un rayon avec le cercle de rayon R."""
    # |pos + t*vel|^2 = R^2  =>  t^2 + 2(pos.vel)t + |pos|^2 - R^2 = 0
    a = np.dot(vel, vel)
    b = 2 * np.dot(pos, vel)
    c = np.dot(pos, pos) - R**2
    disc = b**2 - 4*a*c
    t1 = (-b - np.sqrt(disc)) / (2*a)
    t2 = (-b + np.sqrt(disc)) / (2*a)
    t = t2 if t1 < 1e-10 else t1
    return pos + t * vel

def stade_reflexion(pos, vel, a=0.5, R=0.5):
    """
    Billard de Bunimovich : rectangle [-a,a]x[-R,R] + demi-cercles aux extrémités.
    Retourne (nouveau_pos, nouvelle_vel, t_trajet).
    """
    x, y = pos
    vx, vy = vel
    t_min = np.inf
    type_mur = None

    # murs horizontaux y = ±R (partie rectangulaire |x| <= a)
    for sign in [1, -1]:
        if vy * sign > 0:
            t = (sign * R - y) / vy
            xi = x + t * vx
            if t > 1e-10 and abs(xi) <= a:
                if t < t_min:
                    t_min = t
                    type_mur = ('h', sign)

    # demi-cercles x = ±a + R*cos(theta)
    for sign in [1, -1]:
        cx = sign * a
        dx, dy = x - cx, y
        A = vx**2 + vy**2
        B = 2*(dx*vx + dy*vy)
        C = dx**2 + dy**2 - R**2
        disc = B**2 - 4*A*C
        if disc >= 0:
            for t_cand in [(-B + np.sqrt(disc))/(2*A),
                           (-B - np.sqrt(disc))/(2*A)]:
                if t_cand > 1e-10:
                    xi = x + t_cand * vx
                    if sign * xi >= a:   # dans le bon demi-cercle
                        if t_cand < t_min:
                            t_min = t_cand
                            type_mur = ('c', sign, cx)

    new_pos = pos + t_min * vel
    if type_mur[0] == 'h':
        new_vel = np.array([vx, -vy])
    else:
        cx = type_mur[2]
        n = (new_pos - np.array([cx, 0.0]))
        n /= np.linalg.norm(n)
        new_vel = vel - 2 * np.dot(vel, n) * n

    return new_pos, new_vel

N_rebonds = 1500

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- Billard circulaire
rng = np.random.default_rng(7)
pos = np.array([0.5, 0.0])
angle = rng.uniform(0, 2*np.pi)
vel = np.array([np.cos(angle), np.sin(angle)])
pts = [pos.copy()]
R = 1.0
for _ in range(N_rebonds):
    pos = prochain_cercle(pos, vel, R)
    vel = reflexion_cercle(pos, vel)
    pts.append(pos.copy())
pts = np.array(pts)
axes[0].plot(pts[:, 0], pts[:, 1], 'steelblue', lw=0.15, alpha=0.6)
theta = np.linspace(0, 2*np.pi, 300)
axes[0].plot(np.cos(theta), np.sin(theta), 'k-', lw=1.5)
axes[0].set_aspect('equal')
axes[0].axis('off')
axes[0].set_title('Billard circulaire\n(non ergodique)')

# --- Billard de Bunimovich
a, R_s = 0.5, 0.5
rng2 = np.random.default_rng(3)
pos2 = np.array([rng2.uniform(-a, a), rng2.uniform(-R_s, R_s)])
ang2 = rng2.uniform(0, 2*np.pi)
vel2 = np.array([np.cos(ang2), np.sin(ang2)])
pts2 = [pos2.copy()]
for _ in range(N_rebonds):
    try:
        pos2, vel2 = stade_reflexion(pos2, vel2, a, R_s)
        pts2.append(pos2.copy())
    except Exception:
        break
pts2 = np.array(pts2)
axes[1].plot(pts2[:, 0], pts2[:, 1], 'firebrick', lw=0.15, alpha=0.6)
# contour du stade
t_arc = np.linspace(-np.pi/2, np.pi/2, 100)
axes[1].plot(a + R_s*np.cos(t_arc),  R_s*np.sin(t_arc),  'k-', lw=1.5)
axes[1].plot(-a + R_s*np.cos(t_arc + np.pi), R_s*np.sin(t_arc + np.pi), 'k-', lw=1.5)
axes[1].plot([-a, a], [ R_s,  R_s], 'k-', lw=1.5)
axes[1].plot([-a, a], [-R_s, -R_s], 'k-', lw=1.5)
axes[1].set_aspect('equal')
axes[1].axis('off')
axes[1].set_title('Billard de Bunimovich\n(ergodique)')

plt.tight_layout()
save("ergo_billard")


# ------------------------------------------------------------
# ergo_lyapunov : application logistique, divergence + spectre
# ------------------------------------------------------------
print("ergo_lyapunov + ergo_spectre...")

def logistique(r, x0, N):
    x = np.empty(N)
    x[0] = x0
    for n in range(1, N):
        x[n] = r * x[n-1] * (1.0 - x[n-1])
    return x

def lyapunov(r, N=3000, transient=500):
    x = 0.5
    for _ in range(transient):
        x = r * x * (1.0 - x)
    lam = 0.0
    for _ in range(N):
        x = r * x * (1.0 - x)
        d = abs(r * (1.0 - 2.0 * x))
        if d > 0:
            lam += np.log(d)
    return lam / N

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
configs = [(3.2, r'$r=3{,}2$ (périodique)'),
           (3.9, r'$r=3{,}9$ (chaotique)')]

for col, (r, label) in enumerate(configs):
    x1 = logistique(r, 0.500,  60)
    x2 = logistique(r, 0.5001, 60)
    delta = np.abs(x1 - x2)
    lam = lyapunov(r)

    axes[0, col].plot(x1, color='steelblue', lw=1.2,
                      label=r'$x_0=0{,}500$')
    axes[0, col].plot(x2, color='firebrick', lw=1.2, ls='--',
                      label=r'$x_0=0{,}5001$')
    axes[0, col].set_title(f'{label}\n$\\lambda = {lam:.3f}$')
    axes[0, col].set_ylabel('$x_n$')
    axes[0, col].legend(fontsize=8)
    axes[0, col].set_ylim(-0.05, 1.05)

    axes[1, col].semilogy(delta + 1e-16, color='purple', lw=1.2)
    axes[1, col].set_xlabel('Itération $n$')
    axes[1, col].set_ylabel(r'$|\delta x_n|$')
    axes[1, col].set_ylim(1e-16, 2)

plt.tight_layout()
save("ergo_lyapunov")

# Spectre de Lyapunov
r_vals = np.linspace(2.5, 4.0, 700)
lam_vals = np.array([lyapunov(r) for r in r_vals])

plt.figure(figsize=(8, 4))
plt.plot(r_vals, lam_vals, 'k-', lw=0.7)
plt.axhline(0, color='firebrick', lw=1.5, ls='--')
plt.fill_between(r_vals, lam_vals, 0,
                 where=(lam_vals > 0),
                 color='orange', alpha=0.45,
                 label=r'chaos ($\lambda > 0$)')
plt.xlabel('Paramètre de contrôle $r$ (sans dimension)')
plt.ylabel(r'Exposant de Lyapunov $\lambda$ (sans unité)')
plt.legend()
plt.tight_layout()
save("ergo_spectre")


# ------------------------------------------------------------
# ergo_paysage : paysage d'énergie schématique
# ------------------------------------------------------------
print("ergo_paysage...")

x = np.linspace(-3.5, 3.5, 500)
# paysage avec un minimum global et deux minima locaux
E = (0.8*np.cos(1.8*x) + 0.35*np.cos(3.5*x)
     + 0.15*x**2 - 0.5*np.exp(-0.5*(x+2.8)**2))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, E, 'k-', lw=2)
ax.fill_between(x, E, E.min() - 0.3, alpha=0.08, color='steelblue')

# repères
i_glob = np.argmin(E)
ax.annotate('Minimum global\n(cristal)',
            xy=(x[i_glob], E[i_glob]),
            xytext=(x[i_glob] - 0.9, E[i_glob] + 1.9),
            arrowprops=dict(arrowstyle='->', color='firebrick'),
            color='firebrick', fontsize=9, ha='left')

# minima locaux
from scipy.signal import argrelmin
idx_loc = argrelmin(E, order=20)[0]
for idx in idx_loc:
    if idx != i_glob:
        ax.annotate('Minimum local\n(verre)',
                    xy=(x[idx], E[idx]),
                    xytext=(x[idx] + 1.3, E[idx] + 1.9),
                    arrowprops=dict(arrowstyle='->', color='steelblue'),
                    color='steelblue', fontsize=9, ha='right')
        break

ax.set_xlabel('Coordonnée de configuration (sans dimension)')
ax.set_ylabel('Énergie (unités arbitraires)')
ax.set_title('Paysage d\'énergie : brisure d\'ergodicité')
ax.set_xlim(-3.5, 3.5)
plt.tight_layout()
save("ergo_paysage")


# ============================================================
# CHAPITRE FRACTALES
# ============================================================

# ------------------------------------------------------------
# fractales_koch : construction du flocon de Koch
# ------------------------------------------------------------
print("fractales_koch...")

def subdivise(pts):
    nouveaux = []
    for A, B in zip(pts[:-1], pts[1:]):
        P = A + (B - A) / 3
        Q = A + (B - A) * 2 / 3
        M = P + (Q - P) * np.exp(-1j * np.pi / 3)
        nouveaux += [A, P, M, Q]
    nouveaux.append(pts[-1])
    return nouveaux

a_k = 0+0j
b_k = 1+0j
c_k = 0.5 + np.sqrt(3)/2 * 1j
pts_base = [a_k, b_k, c_k, a_k]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for ax, n in zip(axes, [0, 1, 2, 4]):
    tmp = pts_base.copy()
    for _ in range(n):
        tmp = subdivise(tmp)
    xs = [p.real for p in tmp]
    ys = [p.imag for p in tmp]
    ax.plot(xs, ys, 'k-', lw=0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    perim_ratio = (4/3)**n
    ax.set_title(f'Étape {n}\n$L/L_0 = {perim_ratio:.2f}$', fontsize=10)

plt.suptitle('Construction du flocon de Koch', y=1.02)
plt.tight_layout()
save("fractales_koch")


# ------------------------------------------------------------
# fractales_boxcount : box-counting sur le flocon
# ------------------------------------------------------------
print("fractales_boxcount...")

def points_koch(n_iter=6, n_pts=60000, seed=0):
    rng = np.random.default_rng(seed)
    tmp = pts_base.copy()
    for _ in range(n_iter):
        tmp = subdivise(tmp)
    segs = list(zip(tmp[:-1], tmp[1:]))
    longueurs = np.array([abs(B - A) for A, B in segs])
    longueurs /= longueurs.sum()
    idx = rng.choice(len(segs), size=n_pts, p=longueurs)
    t_arr = rng.random(n_pts)
    pts_arr = np.array(
        [segs[i][0] + t_arr[k] * (segs[i][1] - segs[i][0])
         for k, i in enumerate(idx)]
    )
    return pts_arr.real, pts_arr.imag

def box_counting(xs, ys, eps_vals):
    N_vals = []
    for eps in eps_vals:
        bx = np.floor(xs / eps).astype(int)
        by = np.floor(ys / eps).astype(int)
        N_vals.append(len(set(zip(bx, by))))
    return np.array(N_vals)

xs_k, ys_k = points_koch()
eps_vals = np.logspace(-0.1, -2.0, 25)
N_vals = box_counting(xs_k, ys_k, eps_vals)

log_inv_eps = np.log(1 / eps_vals)
log_N = np.log(N_vals)
coef = np.polyfit(log_inv_eps, log_N, 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# flocon
tmp = pts_base.copy()
for _ in range(5):
    tmp = subdivise(tmp)
xs_f = [p.real for p in tmp]
ys_f = [p.imag for p in tmp]
axes[0].plot(xs_f, ys_f, 'k-', lw=0.6)
axes[0].set_aspect('equal')
axes[0].axis('off')
axes[0].set_title('Flocon de Koch (étape 5)')

# log-log
axes[1].scatter(log_inv_eps, log_N, s=25, color='steelblue', zorder=3,
                label='Données')
axes[1].plot(log_inv_eps, np.polyval(coef, log_inv_eps),
             'r--', lw=1.8,
             label=f'$D \\approx {coef[0]:.3f}$\n(théorie : $\\ln4/\\ln3 \\approx 1{{\,}}262$)')
axes[1].set_xlabel(r'$\ln(1/\varepsilon)$')
axes[1].set_ylabel(r'$\ln N(\varepsilon)$')
axes[1].set_title('Box-counting sur le flocon de Koch')
axes[1].legend()

plt.tight_layout()
save("fractales_boxcount")


# ------------------------------------------------------------
# fractales_lorenz : attracteur de Lorenz
# ------------------------------------------------------------
print("fractales_lorenz...")

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

sol = solve_ivp(lorenz, [0, 60], [0.1, 0, 0],
                method='RK45', max_step=0.01,
                dense_output=False)
# on enlève le transitoire
i0 = int(0.15 * len(sol.t))
x_l, y_l, z_l = sol.y[:, i0:]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].plot(x_l, z_l, 'k-', lw=0.25, alpha=0.7)
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$z$')
axes[0].set_title('Attracteur de Lorenz\n(plan $xz$, vue de face)')

axes[1].plot(x_l, y_l, 'steelblue', lw=0.25, alpha=0.7)
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('$y$')
axes[1].set_title('Attracteur de Lorenz\n(plan $xy$, vue de dessus)')

plt.suptitle(r'$\sigma=10,\;\rho=28,\;\beta=8/3$  —  $D \approx 2{,}06$',
             fontsize=10)
plt.tight_layout()
save("fractales_lorenz")


# ------------------------------------------------------------
# fractales_dla : agrégat DLA
# ------------------------------------------------------------
print("fractales_dla...")

def dla(N_part=4000, taille=300, seed=42):
    rng = np.random.default_rng(seed)
    grille = np.zeros((taille, taille), dtype=bool)
    cx, cy = taille // 2, taille // 2
    grille[cx, cy] = True

    occ = np.array([[cx, cy]])   # positions occupées pour rayon max

    for _ in range(N_part):
        r_max = int(np.max(np.abs(occ - np.array([cx, cy])))) + 12
        r_launch = min(r_max, taille // 2 - 8)
        theta = rng.uniform(0, 2*np.pi)
        x = int(cx + r_launch * np.cos(theta))
        y = int(cy + r_launch * np.sin(theta))

        placed = False
        for _ in range(200000):
            step = rng.integers(0, 4)
            dx, dy = [(1,0),(-1,0),(0,1),(0,-1)][step]
            x += dx;  y += dy
            if abs(x-cx) >= taille//2-2 or abs(y-cy) >= taille//2-2:
                break
            voisins = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
            if any(0 <= nx < taille and 0 <= ny < taille and grille[nx, ny]
                   for nx, ny in voisins):
                grille[x, y] = True
                occ = np.vstack([occ, [x, y]])
                placed = True
                break

    return grille

grille_dla = dla()

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.imshow(~grille_dla, cmap='gray', interpolation='nearest')
ax.axis('off')
ax.set_title('Agrégat DLA — ~4000 particules\n'
             r'Dimension fractale $D \approx 1{,}71$', fontsize=10)
plt.tight_layout()
save("fractales_dla")


print("\nToutes les figures générées.")

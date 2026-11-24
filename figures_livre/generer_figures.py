#!/usr/bin/env python3
"""
generer_figures.py — Génère toutes les figures du livre
"Le second principe et l'entropie"
Sortie : dossier figures/ avec tous les PDF
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import label
from collections import deque, Counter
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
    plt.tight_layout()
    plt.savefig(f"{OUT}/{nom}.pdf", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {nom}.pdf")

# =============================================================
# CHAPITRE 1 — Cycle de Carnot (diagramme T-S et P-V)
# =============================================================
print("\n=== Chapitre 1 : Carnot ===")

# Une mole de gaz parfait monoatomique, sources a Tc et Tf.
# Les deux diagrammes sont superposes pour que le cycle P-V soit large.
Th, Tf = 600.0, 300.0            # K
Rg, gam = 8.314, 5/3
VA, VB = 10.0, 30.0              # L
k = (Th / Tf)**(1 / (gam - 1))   # rapport de volume sur les adiabatiques
VC, VD = VB * k, VA * k

def P_iso(V, T):                 # pression en bar, V en litres
    return Rg * T / (V * 1e-3) / 1e5

def P_adia(V, V0, T0):
    return P_iso(V0, T0) * (V0 / V)**gam

# Les quatre etapes, parcourues dans l'ordre A -> B -> C -> D -> A
etapes = []
for V1, V2, f in [(VA, VB, lambda V: P_iso(V, Th)),
                  (VB, VC, lambda V: P_adia(V, VB, Th)),
                  (VC, VD, lambda V: P_iso(V, Tf)),
                  (VD, VA, lambda V: P_adia(V, VA, Th))]:
    V = np.linspace(V1, V2, 200)
    etapes.append((V, f(V)))

coul = ['firebrick', '0.35', 'royalblue', '0.35']
noms = [r'Isotherme $T_c$', 'Adiabatiques', r'Isotherme $T_f$', None]

fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2),
                         gridspec_kw={'height_ratios': [1.35, 1]})

# --- Diagramme P-V ---
ax = axes[0]
# polygone ferme : seule l'aire INTERIEURE au cycle est coloree
ax.fill(np.concatenate([e[0] for e in etapes]),
        np.concatenate([e[1] for e in etapes]),
        color='limegreen', alpha=0.35, lw=0,
        label=r'Travail fourni $-W = Q_c + Q_f$')
for (V, P), c, nom in zip(etapes, coul, noms):
    ax.plot(V, P, color=c, lw=2.4, label=nom)
    ax.annotate('', xy=(V[112], P[112]), xytext=(V[96], P[96]),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=2,
                                mutation_scale=20))
for V, P, txt, dec in [(VA, P_iso(VA, Th), 'A', (8, 4)),
                       (VB, P_iso(VB, Th), 'B', (8, 5)),
                       (VC, P_iso(VC, Tf), 'C', (4, 9)),
                       (VD, P_iso(VD, Tf), 'D', (-14, -15))]:
    ax.plot(V, P, 'ko', ms=7, zorder=5)
    ax.annotate(txt, (V, P), textcoords='offset points',
                xytext=dec, fontsize=14, fontweight='bold')
ax.annotate(r'$Q_c > 0$', xy=(17.5, P_iso(17.5, Th)), xytext=(25, 3.75),
            color='firebrick', fontsize=14,
            arrowprops=dict(arrowstyle='-|>', color='firebrick', lw=2))
ax.annotate('', xy=(38, 0.12), xytext=(46, P_iso(46, Tf)),
            arrowprops=dict(arrowstyle='-|>', color='royalblue', lw=2))
ax.text(36.5, 0.1, r'$Q_f < 0$', color='royalblue', fontsize=14, ha='right')
ax.text(27, 1.22, '$-W$', fontsize=20, color='darkgreen',
        ha='center', va='center')
ax.set_xlim(0, 92)
ax.set_ylim(0, 5.4)
ax.set_xlabel('Volume $V$ (L)')
ax.set_ylabel('Pression $P$ (bar)')
ax.set_title('Cycle de Carnot : diagramme $P$-$V$')
ax.legend(loc='upper right')

# --- Diagramme T-S ---
ax = axes[1]
dS = Rg * np.log(VB / VA)        # J/K
S1, S2 = 2.0, 2.0 + dS
ax.fill([S1, S2, S2, S1], [Th, Th, Tf, Tf], color='limegreen',
        alpha=0.35, lw=0)
for (x1, y1, x2, y2), c in zip([(S1, Th, S2, Th), (S2, Th, S2, Tf),
                                (S2, Tf, S1, Tf), (S1, Tf, S1, Th)], coul):
    ax.plot([x1, x2], [y1, y2], color=c, lw=2.4)
    ax.annotate('', xy=(x1 + 0.56*(x2-x1), y1 + 0.56*(y2-y1)),
                xytext=(x1 + 0.48*(x2-x1), y1 + 0.48*(y2-y1)),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=2,
                                mutation_scale=20))
for x, y, txt, dec in [(S1, Th, 'A', (-18, 4)), (S2, Th, 'B', (8, 4)),
                       (S2, Tf, 'C', (8, -14)), (S1, Tf, 'D', (-18, -14))]:
    ax.plot(x, y, 'ko', ms=7, zorder=5)
    ax.annotate(txt, (x, y), textcoords='offset points',
                xytext=dec, fontsize=14, fontweight='bold')
Sm = (S1 + S2) / 2
ax.text(Sm, (Th + Tf) / 2, r'$-W = (T_c - T_f)\,\Delta S$', fontsize=15,
        color='darkgreen', ha='center', va='center')
ax.text(Sm, Th + 22, r'$Q_c = T_c\,\Delta S > 0$', color='firebrick',
        fontsize=13, ha='center')
ax.text(Sm, Tf - 52, r'$Q_f = -\,T_f\,\Delta S < 0$', color='royalblue',
        fontsize=13, ha='center')
ax.set_xlim(0, S2 + 2)
ax.set_ylim(200, 700)
ax.set_yticks([Tf, Th])
ax.set_yticklabels(['$T_f$ = 300', '$T_c$ = 600'])
ax.set_xticks([S1, S2])
ax.set_xticklabels(['$S_A = S_D$', '$S_B = S_C$'])
ax.set_xlabel(r'Entropie $S$ (J$\cdot$K$^{-1}$)')
ax.set_ylabel('Température $T$ (K)')
ax.set_title('Cycle de Carnot : diagramme $T$-$S$')

save("carnot_cycles")

# =============================================================
# CHAPITRE 2 — Rendement de Carnot en fonction de Tc/Tf
# =============================================================
print("\n=== Chapitre 2 : rendement ===")

Tf_val = 300  # K fixe
Tc_vals = np.linspace(310, 1200, 200)
eta_vals = 1 - Tf_val / Tc_vals

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Tc_vals, eta_vals * 100, 'b-', lw=2)
ax.axhline(100, color='gray', ls=':', lw=1)
ax.fill_between(Tc_vals, eta_vals*100, 100,
                alpha=0.15, color='red',
                label='Chaleur cédée à la source froide $|Q_f|/Q_c$ (%)')
ax.set_xlabel('Température source chaude $T_c$ (K)')
ax.set_ylabel('Rendement $\\eta$ (%)')
ax.set_title(f'Rendement de Carnot ($T_f = {Tf_val}$ K fixée)')
ax.set_ylim(0, 105)
ax.legend()
ax.text(800, 30,
        '$\\eta = 1 - T_f/T_c$\n'
        '$T_f = 300$ K\n'
        'Unités : K, sans dimension',
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9),
        fontsize=10)
save("rendement_carnot")

# =============================================================
# CHAPITRE 3 — Diffusion et entropie croissante
# =============================================================
print("\n=== Chapitre 3 : diffusion ===")

np.random.seed(0)
N_part, steps = 2000, 400
positions = np.zeros(N_part)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
instants = [0, 10, 50, steps]

pos_snap = {0: positions.copy()}
p = positions.copy()
for t in range(1, steps+1):
    p = p + np.random.choice([-1, 1], size=N_part)
    if t in instants:
        pos_snap[t] = p.copy()

for ax, t in zip(axes, instants):
    ax.hist(pos_snap[t], bins=50, color='steelblue',
            edgecolor='none', density=True)
    # entropie approx = 0.5*ln(2*pi*e*sigma^2)
    sig = pos_snap[t].std() if t > 0 else 0.5
    S_approx = 0.5 * np.log(2 * np.pi * np.e * max(sig**2, 0.01))
    ax.set_title(f'$t = {t}$ pas\n$S \\approx {S_approx:.2f}$ (u.a.)',
                 fontsize=10)
    ax.set_xlim(-100, 100)
    ax.set_xlabel('Position (u.a.)')
    if t == 0:
        ax.set_ylabel('Densité (u.a.$^{-1}$)')

fig.suptitle('Diffusion par marche aléatoire — entropie $S$ croissante',
             fontsize=11)
save("diffusion")

# =============================================================
# CHAPITRE 4 — Boîte de Boltzmann
# =============================================================
print("\n=== Chapitre 4 : Boltzmann ===")

np.random.seed(1)
N_mol = 60
gauche = np.ones(N_mol, dtype=bool)  # toutes à gauche
n_gauche = [N_mol]
S_vals = []

kB = 1.380649e-23  # J/K (constante de Boltzmann)

from math import comb, log
def entropie_boite(n, N):
    """S = kB * ln(C(N,n))  en J/K"""
    if n == 0 or n == N:
        return 0.0
    return kB * log(comb(N, n))

for _ in range(400):
    i = np.random.randint(N_mol)
    gauche[i] = ~gauche[i]
    ng = gauche.sum()
    n_gauche.append(ng)
    S_vals.append(entropie_boite(ng, N_mol))

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axes[0].plot(n_gauche[1:], color='steelblue', lw=1.2)
axes[0].axhline(N_mol//2, ls='--', color='firebrick', lw=1.5,
                label='Équilibre : $n = N/2$')
axes[0].set_ylabel('Particules côté gauche')
axes[0].legend()
axes[0].set_title(f'Boîte de Boltzmann — $N = {N_mol}$ molécules')

axes[1].plot([s/kB for s in S_vals], color='forestgreen', lw=1.2)
axes[1].axhline(entropie_boite(N_mol//2, N_mol)/kB, ls='--',
                color='firebrick', lw=1.5,
                label=f'$S_{{max}} = k_B \\ln C({N_mol},{N_mol//2})$')
axes[1].set_ylabel('$S / k_B$ (sans dimension)')
axes[1].set_xlabel('Nombre de tirages')
axes[1].legend()

fig.suptitle('$S = k_B \\ln \\Omega$ — relaxation vers l\'équilibre\n'
             '$k_B = 1{,}381 \\times 10^{-23}$ J·K$^{-1}$', fontsize=11)
save("boltzmann_boite")

# Histogramme de Omega
fig, ax = plt.subplots(figsize=(7, 4))
n_range = np.arange(0, N_mol+1)
omega = np.array([comb(N_mol, n) for n in n_range], dtype=float)
S_range = np.array([entropie_boite(n, N_mol) for n in n_range])

ax.bar(n_range, S_range/kB, color='steelblue', alpha=0.8, width=0.8)
ax.set_xlabel('Nombre de molécules à gauche $n$')
ax.set_ylabel('$S / k_B = \\ln \\Omega(n)$ (sans dimension)')
ax.set_title('Entropie $S = k_B \\ln \\Omega$ selon la répartition\n'
             '$k_B = 1{,}381 \\times 10^{-23}$ J·K$^{-1}$')
save("boltzmann_omega")

# =============================================================
# CHAPITRE 5 — Entropie de Shannon
# =============================================================
print("\n=== Chapitre 5 : Shannon ===")

def H_shannon(texte):
    """Entropie de Shannon en bits."""
    texte = texte.lower().replace(' ', '')
    counts = np.array(list(Counter(texte).values()), dtype=float)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-15))

exemples = {
    'Français\n(redondant)':
        "la thermodynamique est la science de la chaleur et du travail",
    'Latin\n(moins redondant)':
        "thermodynamica scientia est de calore et opere mechanico",
    'Aléatoire\n(maximal)':
        "xkqzmvbpwjrtlfhgdsnuoaeyixkqzmvbpwjrtlfhg",
    'Un seul\ncaractère':
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
}

H_vals = [H_shannon(t) for t in exemples.values()]
labels = list(exemples.keys())
couleurs_bar = ['steelblue', 'royalblue', 'firebrick', 'gray']

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].barh(labels, H_vals, color=couleurs_bar, alpha=0.85)
axes[0].set_xlabel('Entropie de Shannon $H$ (bits·caractère$^{-1}$)')
axes[0].set_title('Entropie de Shannon selon la source')
axes[0].axvline(np.log2(26), ls='--', color='black', lw=1,
                label=f'Max théo. = $\\log_2(26) \\approx {np.log2(26):.2f}$ bits')
axes[0].legend(fontsize=9)
for i, v in enumerate(H_vals):
    axes[0].text(v+0.05, i, f'{v:.2f} bits', va='center', fontsize=9)

# Distribution des caractères pour le texte français
texte_fr = "la thermodynamique est la science de la chaleur et du travail"
chars = [c for c in texte_fr.lower() if c != ' ']
cnt = Counter(chars)
ch, freq = zip(*sorted(cnt.items(), key=lambda x: -x[1]))
p_ch = np.array(freq, dtype=float) / sum(freq)

axes[1].bar(ch, p_ch, color='steelblue', alpha=0.8)
axes[1].set_xlabel('Caractère')
axes[1].set_ylabel('Fréquence $p_i$ (sans dimension)')
axes[1].set_title('Distribution des caractères — texte français\n'
                  f'$H = {H_shannon(texte_fr):.2f}$ bits·car.$^{{-1}}$')

save("shannon_entropie")

# =============================================================
# CHAPITRE 6 — Structures dissipatives (tourbillon simple)
# =============================================================
print("\n=== Chapitre 6 : vie / structures dissipatives ===")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Flux d'entropie : schéma bilan
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
ax.set_title('Bilan entropique d\'un système vivant')

# Boîtes
for (x, y, w, h, col, txt) in [
    (0.3, 2.0, 2.0, 2.0, 'gold',       'Soleil\n(ordre,\nhaute $T$)'),
    (3.8, 2.0, 2.4, 2.0, 'lightgreen', 'Cellule\n(système\nvivant)'),
    (7.5, 2.0, 2.0, 2.0, 'lightyellow','Milieu\n(chaleur,\nhaute $S$)'),
]:
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.1',
        fc=col, ec='black', lw=1.5))
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center',
            fontsize=9, fontweight='bold')

# Flèches
for (x1,y1,x2,y2,lbl,col) in [
    (2.3,3.0,3.8,3.0,'$Q_{in}>0$\n(J)', 'red'),
    (6.2,3.0,7.5,3.0,'$Q_{out}<0$\n(J)', 'blue'),
]:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.text((x1+x2)/2, y1+0.35, lbl, ha='center',
            color=col, fontsize=9)

ax.text(5.0, 1.2,
        r'$\Delta S_{univers} = \Delta S_{cellule} + \Delta S_{milieu} \geq 0$',
        ha='center', fontsize=9,
        bbox=dict(boxstyle='round', fc='white', ec='gray'))

# Flux d'énergie solaire vs thermique
ax = axes[1]
T_soleil = 5778    # K
T_corps   = 310    # K (corps humain)
Q = 1.0            # W normalisé

S_in  = Q / T_soleil
S_out = Q / T_corps
categories = ['$\\Delta S_{in}$ (J·K$^{-1}$·s$^{-1}$)',
               '$|\\Delta S_{out}|$ (J·K$^{-1}$·s$^{-1}$)']
vals = [S_in * 1e4, S_out * 1e4]  # ×1e4 pour lisibilité

bars = ax.bar(categories, vals,
              color=['gold', 'royalblue'], alpha=0.85, width=0.5)
ax.set_ylabel('Flux d\'entropie $\\times 10^4$ (J·K$^{-1}$·s$^{-1}$·W$^{-1}$)')
ax.set_title(f'La vie exporte l\'entropie\n'
             f'$T_\\odot = {T_soleil}$ K, $T_{{corps}} = {T_corps}$ K')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.002,
            f'{v:.4f}', ha='center', fontsize=9)

save("vie_entropie")

# =============================================================
# CHAPITRE 7 — Percolation (4 figures)
# =============================================================
print("\n=== Chapitre 7 : percolation ===")

def percole_site(N, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    grille = np.random.rand(N, N) < p
    etiq, _ = label(grille)
    haut = set(etiq[0,:]) - {0}
    bas  = set(etiq[-1,:]) - {0}
    return bool(haut & bas), grille, etiq

# Figure 7a : transition
N, n_essais = 100, 40
p_vals = np.linspace(0.4, 0.8, 60)
prob = [np.mean([percole_site(N, p)[0] for _ in range(n_essais)])
        for p in p_vals]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(p_vals, prob, 'o-', color='steelblue', ms=4, lw=1.5)
ax.axvline(0.593, ls='--', color='firebrick', lw=1.5,
           label='$p_c \\approx 0{,}593$')
ax.set_xlabel('Probabilité d\'occupation $p$ (sans dimension)')
ax.set_ylabel('Probabilité de percolation (sans dimension)')
ax.set_title('Transition de percolation — grille $100\\times100$')
ax.legend()
save("percolation_transition")

# Figure 7b : carte des amas
np.random.seed(42)
_, grille300, etiq300 = percole_site(300, 0.593)
n_amas = etiq300.max()
ordre = np.zeros(n_amas+1, dtype=int)
ordre[1:] = np.random.permutation(n_amas) % 20 + 1
image = np.where(grille300, ordre[etiq300], 0).astype(float)
cmap20 = mpl.colormaps['tab20'].resampled(21)
couleurs_amas = ['white'] + [cmap20(i) for i in range(20)]
cmap_amas = ListedColormap(couleurs_amas)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image, cmap=cmap_amas, vmin=0, vmax=20, interpolation='nearest')
ax.axis('off')
ax.set_title(f'Amas à $p_c = 0.593$ — grille $300\\times300$\n'
             f'{n_amas} amas, structure fractale ($d_f \\approx 1.896$)')
save("percolation_carte")

# Figure 7c : percolation de lien (corrosion)
def percole_lien(N, p):
    h = np.random.rand(N, N-1) < p
    v = np.random.rand(N-1, N) < p
    visite = np.zeros((N, N), dtype=bool)
    file = deque()
    for j in range(N):
        visite[0, j] = True
        file.append((0, j))
    while file:
        i, j = file.popleft()
        for di, dj, arr in [(-1,0,v),(1,0,v),(0,-1,h),(0,1,h)]:
            ni, nj = i+di, j+dj
            if 0<=ni<N and 0<=nj<N and not visite[ni,nj]:
                lien = (arr[min(i,ni), min(j,nj)]
                        if arr.shape == (N-1,N) or arr.shape == (N,N-1)
                        else False)
                if di != 0 and 0<=min(i,ni)<N-1:
                    lien = v[min(i,ni), j]
                elif dj != 0 and 0<=min(j,nj)<N-1:
                    lien = h[i, min(j,nj)]
                else:
                    lien = False
                if lien:
                    visite[ni,nj] = True
                    file.append((ni,nj))
    return visite[-1,:].any()

N2, n2 = 80, 30
p2 = np.linspace(0.3, 0.7, 50)
prob2 = [np.mean([percole_lien(N2, p) for _ in range(n2)]) for p in p2]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(p2, prob2, 's-', color='sienna', ms=4, lw=1.5)
ax.axvline(0.5, ls='--', color='firebrick', lw=1.5,
           label='$p_c = 0{,}5$ (exact)')
ax.set_xlabel('Fraction de liaisons ouvertes $p$ (sans dimension)')
ax.set_ylabel('Probabilité de percolation (sans dimension)')
ax.set_title('Corrosion — percolation de lien, grille $80\\times80$')
ax.legend()
save("percolation_lien")

# Figure 7d : feux de forêt
def feu_foret(N, p, graine=None):
    rng = np.random.default_rng(graine)
    foret = (rng.random((N, N)) < p).astype(int)
    file = deque()
    for i in range(N):
        if foret[i,0]==1:
            foret[i,0]=2; file.append((i,0))
    while file:
        i,j = file.popleft()
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni,nj = i+di,j+dj
            if 0<=ni<N and 0<=nj<N and foret[ni,nj]==1:
                foret[ni,nj]=2; file.append((ni,nj))
    return foret

cmap_feu = ListedColormap(['white','forestgreen','firebrick'])
configs = [(0.45,'$p=0{,}45 < p_c$'),(0.593,'$p=p_c\\approx0{,}59$'),
           (0.70,'$p=0{,}70 > p_c$')]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (p, titre) in zip(axes, configs):
    carte = feu_foret(200, p, graine=0)
    ax.imshow(carte, cmap=cmap_feu, vmin=0, vmax=2,
              interpolation='nearest')
    pct = (carte==2).sum()/200**2
    ax.set_title(f'{titre}\n{pct:.0%} brûlé', fontsize=10)
    ax.axis('off')
fig.suptitle('Feux de forêt et percolation — grille $200\\times200$\n'
             '$p$ = fraction d\'arbres (sans dimension)', fontsize=11)
save("feu_foret")

# =============================================================
# CHAPITRE 8 — Démon de Maxwell et coût de Landauer
# =============================================================
print("\n=== Chapitre 8 : démon de Maxwell ===")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Schéma du démon
ax = axes[0]
ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
ax.set_title('Le démon de Maxwell')

# Boîte gauche
ax.add_patch(mpatches.Rectangle((0.5,0.5),3.5,5,
    fc='lightblue', ec='black', lw=2, alpha=0.4))
ax.text(2.25,5.3,'Rapides\n(chaudes)', ha='center', fontsize=9)

# Boîte droite
ax.add_patch(mpatches.Rectangle((6,0.5),3.5,5,
    fc='lightyellow', ec='black', lw=2, alpha=0.4))
ax.text(7.75,5.3,'Lentes\n(froides)', ha='center', fontsize=9)

# Trappe
ax.plot([4,6],[3,3],'k-',lw=3)
ax.add_patch(mpatches.FancyArrowPatch(
    (4,3),(4.8,3), arrowstyle='->', color='black', lw=1.5,
    mutation_scale=15))
ax.text(5,3.3,'Trappe', ha='center', fontsize=9)
ax.text(5,2.5,'Démon', ha='center', fontsize=10, fontstyle='italic')

# Molécules rapides → gauche, lentes → droite
for x,y,col,s in [(1.5,2,'red',r'$\rightarrow$'),
                   (3,4,'red',r'$\rightarrow$'),
                   (7,3,'blue',r'$\leftarrow$'),
                   (8.5,1.5,'blue',r'$\leftarrow$')]:
    ax.text(x, y, s, color=col, fontsize=16, ha='center')

ax.text(5, 0.2,
        r'Effacer 1 bit $\Rightarrow$ $k_B T \ln 2 \approx 2{,}87 \times 10^{-21}$ J à 300 K',
        ha='center', fontsize=8.5,
        bbox=dict(boxstyle='round', fc='lightyellow', ec='gray'))

# Coût énergétique de Landauer en fonction de T
ax = axes[1]
T_range = np.linspace(1, 1000, 200)
kB = 1.380649e-23
E_landauer = kB * T_range * np.log(2)

ax.semilogy(T_range, E_landauer * 1e21, 'b-', lw=2)
ax.axvline(300, ls='--', color='red', lw=1.5,
           label='$T = 300$ K (ambiant)')
ax.axhline(kB*300*np.log(2)*1e21, ls=':', color='gray', lw=1)
ax.set_xlabel('Température $T$ (K)')
ax.set_ylabel('Énergie minimale d\'effacement\n'
              '$E_L = k_B T \\ln 2$ ($\\times 10^{-21}$ J)')
ax.set_title('Principe de Landauer\n'
             '$k_B = 1{,}381 \\times 10^{-23}$ J·K$^{-1}$')
ax.legend()
ax.text(350, 2, f'$E_L(300\\,K) \\approx 2.87 \\times 10^{{-21}}$ J',
        fontsize=9,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))
save("demon_maxwell")

# =============================================================
# CHAPITRE 9 — Univers : entropie des trous noirs
# =============================================================
print("\n=== Chapitre 9 : univers ===")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Entropie de Bekenstein-Hawking vs masse du trou noir
ax = axes[0]
kB  = 1.380649e-23   # J/K
G   = 6.674e-11       # m³·kg⁻¹·s⁻²
c   = 3e8             # m/s
hbar= 1.055e-34       # J·s
M_sol = 1.989e30      # kg

M_range = np.logspace(-1, 2, 200) * M_sol  # 0.1 à 100 masses solaires
# Rayon de Schwarzschild
Rs = 2 * G * M_range / c**2
# Aire de l'horizon
A = 4 * np.pi * Rs**2
# Entropie de Bekenstein-Hawking
S_BH = kB * A * c**3 / (4 * G * hbar)

ax.loglog(M_range / M_sol, S_BH / kB, 'b-', lw=2)
ax.set_xlabel('Masse du trou noir $M$ ($M_\\odot$)')
ax.set_ylabel('Entropie $S_{BH} / k_B$ (sans dimension)')
ax.set_title('Entropie de Bekenstein-Hawking\n'
             r'$S_{BH} = \frac{k_B c^3}{4G\hbar} A$')
ax.text(0.05, 0.95,
        '$S_{BH} \\propto M^2$\n'
        '$\\propto$ aire de l\'horizon',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
ax.grid(True, which='both', alpha=0.3)

# Évolution schématique de l'entropie de l'univers
ax = axes[1]
t = np.linspace(0, 13.8, 300)  # milliards d'années
# Schématique : croissance avec quelques jalons
S_schematic = t**0.8 * 1e88

ax.plot(t, S_schematic, 'k-', lw=2)
jalons = {
    0.0004: ('Recombinaison\n(380 000 ans)', 'right'),
    1.0:    ('Premières\nétoiles', 'left'),
    4.5:    ('Soleil\nformé', 'left'),
    13.8:   ('Aujourd\'hui', 'left'),
}
for t_j, (txt, ha) in jalons.items():
    idx = np.argmin(np.abs(t - t_j))
    ax.plot(t[idx], S_schematic[idx], 'ro', ms=7, zorder=5)
    ax.annotate(txt, (t[idx], S_schematic[idx]),
                textcoords='offset points',
                xytext=(10 if ha=='left' else -10, 0),
                ha=ha, fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax.set_xlabel('Temps (milliards d\'années = Ga)')
ax.set_ylabel('Entropie $S$ (u.a. schématique)')
ax.set_title('Évolution schématique de l\'entropie de l\'Univers\n'
             '$S$ croissante conformément au 2$^e$ principe')
ax.set_xlim(-0.5, 15)
save("univers_entropie")

# =============================================================
# FIGURE CHRONOLOGIE (pour pages liminaires)
# =============================================================
print("\n=== Chronologie ===")

personnages = [
    (1736, 'Watt'),
    (1796, 'Carnot'),
    (1822, 'Clausius'),
    (1824, 'Carnot\n(Réflexions)'),
    (1844, 'Boltzmann'),
    (1865, 'Clausius\n(entropie)'),
    (1867, 'Maxwell\n(démon)'),
    (1877, 'Boltzmann\n($S=k\\ln\\Omega$)'),
    (1906, 'Boltzmann †'),
    (1916, 'Shannon'),
    (1929, 'Szilard'),
    (1944, 'Schrödinger\n(vie?)'),
    (1948, 'Shannon\n(théorie info)'),
    (1961, 'Landauer'),
    (1977, 'Prigogine\n(Nobel)'),
]

fig, ax = plt.subplots(figsize=(13, 3))
ax.axhline(0, color='black', lw=1.5)
ax.set_xlim(1720, 2000)
ax.set_ylim(-2.5, 2.5)
ax.axis('off')
ax.set_title('Frise chronologique — les bâtisseurs de la thermodynamique',
             fontsize=12)

for i, (annee, nom) in enumerate(personnages):
    y = 1.2 if i % 2 == 0 else -1.2
    ax.plot([annee, annee], [0, y*0.7], 'k-', lw=0.8)
    ax.plot(annee, 0, 'o', color='steelblue', ms=6, zorder=5)
    ax.text(annee, y, nom, ha='center', va='bottom' if y>0 else 'top',
            fontsize=7.5, rotation=0)
    ax.text(annee, -y*0.05, str(annee), ha='center',
            va='top' if y>0 else 'bottom', fontsize=6.5, color='gray')

save("chronologie")

print("\n=== Toutes les figures générées ! ===")
import os
pdfs = [f for f in os.listdir(OUT) if f.endswith('.pdf')]
print(f"Total : {len(pdfs)} fichiers PDF dans {OUT}/")

# ==============================================================
# NOUVELLES FIGURES v2
# ==============================================================

# Figure : Von Neumann qubit
print("\n=== Von Neumann qubit ===")

def S_VN_bits(rho):
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return -np.sum(vals * np.log2(vals))

p_vals = np.linspace(0, 1, 200)
S_vals = [S_VN_bits(np.array([[p,0],[0,1-p]])) for p in p_vals]
theta_vals = np.linspace(0, np.pi/2, 100)
S_int = [S_VN_bits(np.array([[np.cos(t)**2,0],[0,np.sin(t)**2]]))
         for t in theta_vals]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(p_vals, S_vals, color='steelblue', lw=2)
axes[0].axvline(0.5, ls='--', color='firebrick', lw=1.5,
                label='Melange maximal')
axes[0].set_xlabel('Probabilite $p$ de $|0\\rangle$')
axes[0].set_ylabel('$S_{VN}$ (bits)')
axes[0].set_title('Entropie de Von Neumann d\'un qubit')
axes[0].legend()

axes[1].plot(np.degrees(theta_vals), S_int, color='forestgreen', lw=2)
axes[1].axvline(45, ls='--', color='firebrick', lw=1.5, label='Paire EPR')
axes[1].set_xlabel('Angle $\\theta$ (degres)')
axes[1].set_ylabel('Entropie d\'intrication (bits)')
axes[1].set_title('Intrication d\'une paire EPR partielle')
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/von_neumann_qubit.pdf", dpi=150, bbox_inches='tight')
plt.close()
print("  OK von_neumann_qubit.pdf")

# Figure : graphes aleatoires
print("\n=== Graphes aleatoires ===")
try:
    import networkx as nx
    n = 300
    c_vals2 = np.linspace(0, 4, 50)
    taille_geante = []
    for c in c_vals2:
        G = nx.erdos_renyi_graph(n, c/n, seed=42)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        taille_geante.append(len(comps[0])/n if comps else 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c_vals2, taille_geante, 'o-', color='steelblue', ms=3, lw=1.5)
    ax.axvline(1.0, ls='--', color='firebrick', lw=1.5,
               label=r'$\langle k\rangle = 1$ (seuil)')
    ax.set_xlabel(r'Degre moyen $\langle k \rangle$')
    ax.set_ylabel('Fraction dans la composante geante')
    ax.set_title(f'Transition de phase dans $G(n,p)$, $n={n}$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/graphes_aleatoires.pdf", dpi=150, bbox_inches='tight')
    plt.close()
    print("  OK graphes_aleatoires.pdf")
except ImportError:
    print("  SKIP (networkx non installe)")

print("\n=== Figures v2 terminees ===")

# =============================================================
# NOUVELLES FIGURES v3 — chapitres d'application
# =============================================================

print("\n=== Gastronomie : dénaturation et entropie aromatique ===")

# --- Courbe de dénaturation de l'albumine ---
DH = 450e3      # J/mol
DS = 1350       # J/mol/K
T_denat_arr = np.linspace(273, 373, 200)
DG = DH - T_denat_arr * DS

fig, axes_g = plt.subplots(1, 2, figsize=(11, 4))

ax = axes_g[0]
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.plot(T_denat_arr - 273, DG / 1e3, color='steelblue', lw=2,
        label=r'$\Delta G = \Delta H - T\,\Delta S$')
T_d = DH / DS
ax.axvline(T_d - 273, color='firebrick', lw=1.5, ls='--',
           label=f'$T_{{dén}} = {T_d - 273:.0f}$°C')
ax.fill_between(T_denat_arr - 273, DG / 1e3, 0,
                where=(DG < 0), alpha=0.15, color='firebrick',
                label='Dénaturation spontanée')
ax.set_xlabel('Température (°C)')
ax.set_ylabel('$\\Delta G$ (kJ/mol)')
ax.set_title('Dénaturation de l\'albumine\n'
             r'$\Delta H = 450$ kJ/mol, $\Delta S = 1350$ J/(mol·K)')
ax.legend(fontsize=9)

# --- Profil aromatique et entropie de Shannon ---
ax = axes_g[1]
familles = ['Fruité', 'Épicé', 'Boisé', 'Floral', 'Minéral', 'Animal']
vin_complex = np.array([0.35, 0.20, 0.15, 0.15, 0.10, 0.05])
vin_simple  = np.array([0.70, 0.20, 0.10, 0,    0,    0   ])

def H_shan_arr(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

H_c = H_shan_arr(vin_complex)
H_s = H_shan_arr(vin_simple)

x_arr = np.arange(len(familles))
w_bar = 0.35
ax.bar(x_arr - w_bar/2, vin_complex, w_bar,
       label=f'Complexe  H={H_c:.2f} bits', color='steelblue', alpha=0.8)
ax.bar(x_arr + w_bar/2, vin_simple, w_bar,
       label=f'Simple    H={H_s:.2f} bit', color='salmon', alpha=0.8)
ax.set_xticks(x_arr)
ax.set_xticklabels(familles, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Fraction aromatique (sans dim.)')
ax.set_title('Entropie aromatique de deux vins\n'
             r'$H = -\sum p_i \log_2 p_i$')
ax.legend(fontsize=9)

save("gastronomie_entropie")

# =============================================================
print("\n=== Bureaucratie : croissance et entropie lexicale ===")

t_b = np.arange(0, 60)
lam_b = 0.025
N0_b = 10_000
N_b = N0_b * np.exp(lam_b * t_b)
mu_b = 0.20
J_b = (mu_b / lam_b) * N0_b * (np.exp(lam_b * t_b) - 1)

fig, axes_b = plt.subplots(1, 2, figsize=(11, 4))

ax = axes_b[0]
ax.semilogy(1970 + t_b, N_b, color='steelblue', lw=2, label='Articles de loi')
ax.semilogy(1970 + t_b, J_b + 1, color='firebrick', lw=2, ls='--',
            label='Jurisprudence cumulée')
ax.semilogy(1970 + t_b, N_b + J_b + 1, color='forestgreen', lw=1.5,
            ls=':', label='Total')
ax.set_xlabel('Année')
ax.set_ylabel('Nombre (échelle log)')
ax.set_title(f'Croissance normative\n'
             r'$N(t)=N_0 e^{\lambda t}$, $\lambda=2{,}5\,\%$/an')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

ax = axes_b[1]
V_loi_b = 300
freq_loi_b = np.ones(V_loi_b)
freq_loi_b[:3] = 150
p_loi_b = freq_loi_b / freq_loi_b.sum()

V_peda_b = 200
r_peda_b = np.arange(1, V_peda_b + 1)
p_peda_b = (1.0 / r_peda_b)
p_peda_b /= p_peda_b.sum()

def H_bits_arr(p):
    p2 = p[p > 1e-15]
    return -np.sum(p2 * np.log2(p2))

H_loi = H_bits_arr(p_loi_b)
H_peda = H_bits_arr(p_peda_b)

ax.plot(np.sort(p_loi_b)[::-1][:100], color='steelblue', lw=1.8,
        label=f'Loi technique  H={H_loi:.1f} bits')
ax.plot(np.sort(p_peda_b)[::-1][:100], color='forestgreen', lw=1.8,
        label=f'Texte pédagogique  H={H_peda:.1f} bits')
ax.set_xlabel('Rang du mot (100 premiers)')
ax.set_ylabel('Probabilité (sans dim.)')
ax.set_title('Distributions lexicales comparées\n'
             r'$H = -\sum p_w \log_2 p_w$')
ax.legend(fontsize=9)

save("bureaucratie_entropie")

# =============================================================
print("\n=== Jeux : entropie de position et bluff optimal ===")

fig, axes_j = plt.subplots(1, 2, figsize=(11, 4))

phases = np.linspace(0, 1, 200)
n_coups = 5 + 30 * 4 * phases * (1 - phases)
n_eff = 1 + (n_coups - 1) * 0.35
H_pos = np.log2(np.maximum(n_eff, 1))

ax = axes_j[0]
ax.plot(phases, n_coups, color='steelblue', lw=2, label='Coups légaux $n$')
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

p_bluff = np.linspace(0.001, 0.999, 200)
H_bluff = -(p_bluff * np.log2(p_bluff)
            + (1 - p_bluff) * np.log2(1 - p_bluff))
ax = axes_j[1]
ax.plot(p_bluff, H_bluff, color='gray', lw=1.5, ls='--',
        label='Entropie max ($p=0{,}5$)')
configs_j = [(10, 10, 'b=10, c=10'), (10, 30, 'b=10, c=30'),
             (10, 90, 'b=10, c=90')]
for b_j, c_j, label_j in configs_j:
    p_opt = b_j / (b_j + c_j)
    H_opt = -(p_opt * np.log2(p_opt) + (1 - p_opt) * np.log2(1 - p_opt))
    ax.plot(p_opt, H_opt, 'o', ms=9, label=f'{label_j}  $p^*$={p_opt:.2f}')
ax.set_xlabel('Fréquence de bluff $p$ (sans dim.)')
ax.set_ylabel('$H^*$ (bits)')
ax.set_title('Entropie de la stratégie de bluff\nà l\'équilibre de Nash')
ax.legend(fontsize=9)

save("echecs_entropie")

print("\n=== Figures v3 terminées ===")

"""L'entropie sous toutes ses formes -- chapitre « S ne fait que croître ».

Mosaique de carres de quatre couleurs : a chaque tirage, deux carres voisins
echangent leurs places. On mesure l'entropie de Shannon des couleurs, bloc par
bloc, puis on rejoue tous les echanges a l'envers.
Option : python mosaique.py --gif  (cree aussi figures/mosaique.gif)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

L, BLOC, BALAYAGES = 48, 8, 1500     # cote, cote d'un bloc, duree
CMAP = ListedColormap(["#c0392b", "#f1c40f", "#2471a3", "#27ae60"])
rng = np.random.default_rng(0)

# etat initial : un quart de la surface par couleur
grille = np.zeros((L, L), dtype=int)
grille[:L//2, L//2:] = 1
grille[L//2:, :L//2] = 2
grille[L//2:, L//2:] = 3


def entropie_blocs(g):
    """Entropie moyenne des couleurs dans les blocs (bits par carre)."""
    b = g.reshape(L//BLOC, BLOC, L//BLOC, BLOC).swapaxes(1, 2)
    b = b.reshape(-1, BLOC**2)
    p = np.stack([(b == k).mean(axis=1) for k in range(4)])
    q = np.where(p > 0, p, 1.0)
    return float(abs((p * np.log2(q)).sum(axis=0).mean()))


# tous les echanges sont tires d'avance : on pourra les rejouer a l'envers
n = BALAYAGES * L * L
i, j = rng.integers(0, L, (2, n))
sens = rng.integers(0, 2, n)
i2, j2 = i + (sens == 0), j + (sens == 1)
garde = (i2 < L) & (j2 < L)
A = (i * L + j)[garde].tolist()
B = (i2 * L + j2)[garde].tolist()
pas = len(A) // BALAYAGES            # echanges par unite de temps

g = grille.ravel().tolist()
cliches = {0: grille.copy()}
H, images = [entropie_blocs(grille)], [grille.copy()]
for t in range(1, BALAYAGES + 1):
    for k in range((t - 1) * pas, t * pas):
        a, b = A[k], B[k]
        g[a], g[b] = g[b], g[a]
    etat = np.array(g).reshape(L, L)
    H.append(entropie_blocs(etat))
    if t in (20, 200, BALAYAGES):
        cliches[t] = etat
    if t % 10 == 0:
        images.append(etat)

# le film a l'envers : memes echanges, ordre inverse
Hr = [H[-1]]
for t in range(BALAYAGES, 0, -1):
    for k in range(t * pas - 1, (t - 1) * pas - 1, -1):
        a, b = A[k], B[k]
        g[a], g[b] = g[b], g[a]
    Hr.append(entropie_blocs(np.array(g).reshape(L, L)))
print("retour exact :", np.array_equal(np.array(g).reshape(L, L), grille))

# entropie d'un melange parfait, meme effectif de chaque couleur
ref = np.mean([entropie_blocs(rng.permutation(grille.ravel()).reshape(L, L))
               for _ in range(400)])
print(f"H(0) = {H[0]:.3f}, H(20) = {H[20]:.2f}, H(200) = {H[200]:.2f}, "
      f"H finale = {np.mean(H[-200:]):.3f}, melange parfait = {ref:.3f}")

# --- figure ---
fig = plt.figure(figsize=(11, 7))
for n_ax, t in enumerate(sorted(cliches)):
    ax = fig.add_subplot(2, 4, n_ax + 1)
    ax.imshow(cliches[t], cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"$t$ = {t}\n$H$ = {H[t]:.2f} bit".replace(".", ","),
                 fontsize=11)
ax = fig.add_subplot(2, 2, 3)
ax.plot(H, color="firebrick", lw=1.3)
ax.axhline(ref, ls="--", color="gray", label="mélange parfait")
ax.set_xlabel("Temps $t$ (échanges par carré)")
ax.set_ylabel("$H$ (bits par carré)")
ax.set_title("Échanges au hasard")
ax.legend(loc="lower right")
ax = fig.add_subplot(2, 2, 4)
ax.plot(BALAYAGES + np.arange(BALAYAGES + 1), Hr, color="steelblue", lw=1.3)
ax.set_xlabel("Temps $t$")
ax.set_ylabel("$H$ (bits par carré)")
ax.set_title("Les mêmes échanges, rejoués à l'envers")
plt.tight_layout()
plt.savefig(FIG / "mosaique.pdf", dpi=200)

if "--gif" in sys.argv:
    from PIL import Image
    pal = (np.array([CMAP(k)[:3] for k in range(4)]) * 255).astype("uint8")
    film = [Image.fromarray(pal[im].repeat(8, 0).repeat(8, 1)) for im in images]
    film[0].save(FIG / "mosaique.gif", save_all=True, append_images=film[1:],
                 duration=60, loop=0)

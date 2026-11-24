"""L'entropie sous toutes ses formes -- chapitre « Réseaux et graphes aléatoires: l'entropie des connexions »."""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


n = 500  # sommets
c_vals = np.linspace(0, 3, 60)  # <k> = c
taille_geante = []

for c in c_vals:
    p = c / n
    G = nx.erdos_renyi_graph(n, p, seed=42)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    taille_geante.append(len(comps[0]) / n if comps else 0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Courbe de transition
axes[0].plot(c_vals, taille_geante, 'o-', color='steelblue',
             ms=3, lw=1.5)
axes[0].axvline(1.0, ls='--', color='firebrick', lw=1.5,
                label=r'$\langle k\rangle = 1$ (seuil)')
axes[0].set_xlabel(r'Degré moyen $\langle k \rangle$ (sans dim.)')
axes[0].set_ylabel('Fraction dans la composante géante')
axes[0].set_title('Transition de phase dans $G(n,p)$\n'
                  f'$n = {n}$ sommets')
axes[0].legend()

# Entropie spectrale vs <k>
def entropie_spectrale(G):
    L = nx.normalized_laplacian_matrix(G).toarray()
    vals = np.linalg.eigvalsh(L)
    vals = vals[vals > 1e-10]
    vals /= vals.sum()
    return -np.sum(vals * np.log(vals))

c_sparse = np.linspace(0.1, 5, 20)
S_graph = []
for c in c_sparse:
    G = nx.erdos_renyi_graph(n, c/n, seed=0)
    S_graph.append(entropie_spectrale(G))

axes[1].plot(c_sparse, S_graph, 's-', color='forestgreen',
             ms=5, lw=1.5)
axes[1].axvline(1.0, ls='--', color='firebrick', lw=1.5,
                label=r'Seuil $\langle k\rangle = 1$')
axes[1].set_xlabel(r'Degré moyen $\langle k \rangle$ (sans dim.)')
axes[1].set_ylabel('Entropie spectrale $S_{graph}$ (nats)')
axes[1].set_title('Entropie du graphe vs connectivité')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG / 'graphes_aleatoires.pdf', dpi=150)

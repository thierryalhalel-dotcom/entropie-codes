"""L'entropie sous toutes ses formes -- chapitre « Entropie et propagation virale: quand l'information se répand »."""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def sir_network(G, beta=0.3, gamma=0.05, seed=42):
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    state = np.zeros(N, dtype=int)  # 0=S, 1=I, 2=R
    patient_zero = rng.integers(N)
    state[patient_zero] = 1

    history = []
    H_history = []

    for _ in range(200):
        new_state = state.copy()
        for u in G.nodes():
            if state[u] == 1:
                for v in G.neighbors(u):
                    if state[v] == 0 and rng.random() < beta:
                        new_state[v] = 1
                if rng.random() < gamma:
                    new_state[u] = 2
        state = new_state
        counts = np.bincount(state, minlength=3) / N
        history.append(counts.copy())
        p = counts[counts > 0]
        H_history.append(entropy(p, base=2))
        if state.sum() == 2 * N:
            break

    return np.array(history), H_history

# Comparaison réseau aléatoire vs réseau sans-échelle
G_random = nx.erdos_renyi_graph(500, 0.02, seed=0)
G_sf = nx.barabasi_albert_graph(500, 5, seed=0)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, (G, title) in enumerate(
        [(G_random, 'Réseau aléatoire'), (G_sf, 'Réseau sans-échelle')]):
    hist, H_hist = sir_network(G)
    labels = ['S', 'I', 'R']
    colors = ['steelblue', 'tomato', 'seagreen']
    for k, (lab, col) in enumerate(zip(labels, colors)):
        axes[row, 0].plot(hist[:, k], label=lab, color=col)
    axes[row, 0].set_title(f'{title} : SIR'); axes[row, 0].legend()
    axes[row, 0].set_xlabel('Temps'); axes[row, 0].set_ylabel('Fraction')
    axes[row, 1].plot(H_hist, color='purple')
    axes[row, 1].set_title(f'{title} : entropie de la cascade')
    axes[row, 1].set_xlabel('Temps'); axes[row, 1].set_ylabel('bits')

plt.tight_layout()
plt.savefig(FIG / 'sir_cascade_entropie.pdf')

"""L'entropie sous toutes ses formes -- chapitre « Entropie et politique: la polarisation comme transition de phase »."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def deffuant(N=1000, epsilon=0.25, mu=0.5, steps=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, N)
    history = [x.copy()]
    for _ in range(steps * N):
        i, j = rng.choice(N, 2, replace=False)
        if abs(x[i] - x[j]) < epsilon:
            avg = (x[i] + x[j]) / 2
            x[i] += mu * (avg - x[i])
            x[j] += mu * (avg - x[j])
        if _ % (N * 100) == 0:
            history.append(x.copy())
    return history

def opinion_entropy(x, bins=50):
    counts, _ = np.histogram(x, bins=bins, range=(-1, 1), density=False)
    counts = counts + 1e-10
    probs = counts / counts.sum()
    return entropy(probs, base=2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, eps in zip(axes, [0.4, 0.25, 0.1]):
    history = deffuant(epsilon=eps)
    ax.hist(history[-1], bins=50, range=(-1, 1), color='steelblue', alpha=0.7)
    H = opinion_entropy(history[-1])
    ax.set_title(f'$\\varepsilon={eps}$, $H={H:.2f}$ bits')
    ax.set_xlabel('Opinion'); ax.set_ylabel('Effectif')
plt.tight_layout()
plt.savefig(FIG / 'deffuant_polarisation.pdf')

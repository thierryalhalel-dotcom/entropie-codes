"""L'entropie sous toutes ses formes -- chapitre « Entropie et inégalité »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


rng = np.random.default_rng(0)
N = 50_000

def lorenz_curve(vals):
    """Renvoie (fracs_pop, fracs_revenu) pour la courbe de Lorenz."""
    v = np.sort(vals)
    cum = np.cumsum(v)
    fp = np.linspace(0, 1, N + 1)
    fr = np.concatenate([[0.0], cum / cum[-1]])
    return fp, fr

def gini_coeff(vals):
    v = np.sort(vals)
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)

def shannon_H(vals, K=50):
    """Entropie des parts de revenu de K groupes de même effectif."""
    parts = np.sort(vals).reshape(K, -1).sum(axis=1)
    p = parts / parts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

# ------- Générer les distributions -------
# Loi uniforme sur [1, 10]
unif = rng.uniform(1, 10, N)

# Loi exponentielle (MaxEnt) : mu = 4
expo = rng.exponential(scale=4.0, size=N)

# Loi de Pareto (alpha = 1.3) : f(x) ~ x^{-2.3} pour x >= 1
alpha = 1.3
paret = (1.0 / rng.uniform(0, 1, N)) ** (1 / alpha)   # inverse CDF

dists  = {
    'Uniforme':              unif,
    'Exponentielle (MaxEnt)': expo,
    r'Pareto ($\alpha=1{,}3$)':  paret,
}
colors = ['steelblue', 'forestgreen', 'firebrick']

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], 'k--', lw=1.0, label='Égalité parfaite')

for (name, vals), color in zip(dists.items(), colors):
    fp, fr = lorenz_curve(vals)
    G = gini_coeff(vals)
    H = shannon_H(vals)
    ax.plot(fp, fr, lw=2.0, color=color,
            label=f'{name} : Gini $= {G:.3f}$, $H = {H:.1f}$ bits')

ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color='gray')
ax.set_xlabel('Fraction cumulée de la population')
ax.set_ylabel('Fraction cumulée du revenu')
ax.set_title("Courbes de Lorenz et mesures d'inégalité")
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(FIG / 'inegalite_lorenz.pdf', dpi=150)

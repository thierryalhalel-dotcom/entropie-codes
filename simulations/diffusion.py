"""L'entropie sous toutes ses formes -- chapitre « S ne fait que croître »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


np.random.seed(0)
N, steps = 2000, 400           # particules, pas de temps
pos = np.zeros(N)              # toutes en x=0 à t=0

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
instants = [0, 10, 50, steps]
snaps = {0: pos.copy()}

for t in range(1, steps + 1):
    pos += np.random.choice([-1, 1], size=N)  # déplacement vectorisé
    if t in instants:
        snaps[t] = pos.copy()

for ax, t in zip(axes, instants):
    sig = snaps[t].std() if t > 0 else 0.5
    # Entropie de la gaussienne : S = (1/2)ln(2pie sigma**2)
    S = 0.5 * np.log(2 * np.pi * np.e * max(sig**2, 0.01))
    ax.hist(snaps[t], bins=50, density=True,
            color='steelblue', edgecolor='none')
    ax.set_title(f't = {t}\nS ~= {S:.2f} (u.a.)', fontsize=10)
    ax.set_xlim(-100, 100)
    ax.set_xlabel('Position (u.a.)')
axes[0].set_ylabel('Densité (u.a.-1)')

plt.tight_layout()
plt.savefig(FIG / 'diffusion.pdf', dpi=150)

"""L'entropie sous toutes ses formes -- chapitre « Boltzmann: le désordre se compte »."""
import numpy as np
import matplotlib.pyplot as plt
from math import comb, log
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


kB = 1.380649e-23   # J/K --- constante de Boltzmann

def S_boite(n, N):
    """Entropie S = kB * ln(C(N,n)) en J/K."""
    if n == 0 or n == N:
        return 0.0
    return kB * log(comb(N, n))

np.random.seed(1)
N = 60
cote_gauche = np.ones(N, dtype=bool)   # toutes à gauche
n_gauche = [N]
S_vals   = []

for _ in range(400):
    i = np.random.randint(N)            # molécule aléatoire
    cote_gauche[i] = ~cote_gauche[i]   # change de côté
    ng = cote_gauche.sum()
    n_gauche.append(ng)
    S_vals.append(S_boite(ng, N))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(n_gauche[1:], color='steelblue', lw=1.2)
ax1.axhline(N//2, ls='--', color='firebrick', lw=1.5,
            label=f'Équilibre : n = {N//2}')
ax1.set_ylabel('Molécules côté gauche')
ax1.legend()

ax2.plot([s / kB for s in S_vals], color='forestgreen', lw=1.2)
S_max = S_boite(N//2, N) / kB
ax2.axhline(S_max, ls='--', color='firebrick', lw=1.5,
            label=f'$S_{{max}}/k_B = {S_max:.2f}$')
ax2.set_ylabel('$S / k_B$ (sans dimension)')
ax2.set_xlabel('Nombre de tirages')
ax2.legend()

fig.suptitle(
    f'Boîte de Boltzmann --- N = {N} molécules\n'
    r'$S = k_B \ln \Omega$, $k_B = 1{,}381 \times 10^{-23}$ J/K')
plt.tight_layout()
plt.savefig(FIG / 'boltzmann_boite.pdf', dpi=150)

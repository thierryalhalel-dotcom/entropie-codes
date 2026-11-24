"""L'entropie sous toutes ses formes -- chapitre « L'entropie dans la cellule: repliement, ADN et évolution »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


# Sequence HP : H=1, P=0
sequence = [1,0,1,1,0,1,0,0,1,1,0,1,0,1,1,0]
N = len(sequence)
kBT = 1.0

def energie(pos, seq):
    """Energie = -nombre de contacts H-H non consecutifs."""
    E = 0
    for i in range(len(seq)):
        for j in range(i+2, len(seq)):
            if seq[i]==1 and seq[j]==1:
                d = abs(pos[i,0]-pos[j,0]) + abs(pos[i,1]-pos[j,1])
                if d == 1:
                    E -= 1
    return E

def generer_marche(N):
    """Marche aleatoire auto-evitante sur reseau carre."""
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    pos = np.zeros((N,2), dtype=int)
    occupe = {(0,0)}
    for i in range(1, N):
        np.random.shuffle(directions)
        place = False
        for d in directions:
            np_pos = (pos[i-1,0]+d[0], pos[i-1,1]+d[1])
            if np_pos not in occupe:
                pos[i] = [np_pos[0], np_pos[1]]
                occupe.add(np_pos)
                place = True; break
        if not place:
            return None  # echec
    return pos

# Monte Carlo
n_steps = 2000
energies = []
pos = None
while pos is None:
    pos = generer_marche(N)

E_courant = energie(pos, sequence)
for _ in range(n_steps):
    pos_new = generer_marche(N)
    if pos_new is None: continue
    E_new = energie(pos_new, sequence)
    dE = E_new - E_courant
    if dE < 0 or np.random.rand() < np.exp(-dE/kBT):
        pos = pos_new; E_courant = E_new
    energies.append(E_courant)

plt.figure(figsize=(6,4))
plt.plot(energies, color='steelblue', lw=1, alpha=0.8)
plt.xlabel('Etapes Monte Carlo')
plt.ylabel('Energie (unite $k_BT$, sans dim.)')
plt.title('Repliement HP par Monte Carlo\n'
          'Minimisation de $\\Delta G = \\Delta H - T\\Delta S$')
plt.tight_layout()
plt.savefig(FIG / 'repliement_hp.pdf', dpi=150)

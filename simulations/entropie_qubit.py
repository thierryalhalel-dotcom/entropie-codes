"""L'entropie sous toutes ses formes -- chapitre « L'entropie quantique: de Von Neumann aux qubits »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def rho_qubit(p):
    """Matrice densite d'un qubit: melange de |0> (prob p) et |1> (prob 1-p)."""
    return np.array([[p, 0], [0, 1-p]])

def S_VN_bits(rho):
    """Entropie de Von Neumann en bits."""
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return -np.sum(vals * np.log2(vals))

p_vals = np.linspace(0, 1, 200)
S_vals = [S_VN_bits(rho_qubit(p)) for p in p_vals]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(p_vals, S_vals, color='steelblue', lw=2)
axes[0].axvline(0.5, ls='--', color='firebrick', lw=1.5,
                label='Melange maximal ($p=0.5$)')
axes[0].set_xlabel('Probabilite $p$ de $|0\\rangle$ (sans dim.)')
axes[0].set_ylabel('$S_{VN}$ (bits)')
axes[0].set_title('Entropie de Von Neumann d\'un qubit\n'
                  '$S_{VN} = -\\mathrm{Tr}(\\rho \\log_2 \\rho)$')
axes[0].legend()

# Entropie d'intrication d'une paire EPR partielle
theta_vals = np.linspace(0, np.pi/2, 100)
S_int = []
for theta in theta_vals:
    # |psi> = cos(theta)|00> + sin(theta)|11>
    # rho_A = [[cos^2, 0],[0, sin^2]]
    rho_A = np.array([[np.cos(theta)**2, 0],
                       [0, np.sin(theta)**2]])
    S_int.append(S_VN_bits(rho_A))

axes[1].plot(np.degrees(theta_vals), S_int,
             color='forestgreen', lw=2)
axes[1].axvline(45, ls='--', color='firebrick', lw=1.5,
                label='Paire EPR ($\\theta = 45^\\circ$)')
axes[1].set_xlabel('Angle $\\theta$ (degres)')
axes[1].set_ylabel('Entropie d\'intrication $S_{int}$ (bits)')
axes[1].set_title('Entropie d\'intrication\n'
                  '$|\\Psi\\rangle = \\cos\\theta|00\\rangle '
                  '+ \\sin\\theta|11\\rangle$')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG / 'von_neumann_qubit.pdf', dpi=150)

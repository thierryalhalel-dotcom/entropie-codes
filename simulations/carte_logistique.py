"""L'entropie sous toutes ses formes -- chapitre « Entropie et ergodicité »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def logistique(r, x0, N):
    """Itere N fois la carte logistique depuis x0."""
    x = np.empty(N)
    x[0] = x0
    for n in range(1, N):
        x[n] = r * x[n-1] * (1.0 - x[n-1])
    return x

def lyapunov(r, N=3000, transient=500):
    """Exposant de Lyapunov par la formule log|f'(x)|."""
    x = 0.5
    for _ in range(transient):          # elimine le transitoire
        x = r * x * (1.0 - x)
    lam = 0.0
    for _ in range(N):
        x = r * x * (1.0 - x)
        d = abs(r * (1.0 - 2.0 * x))   # |f'(x)|
        if d > 0:
            lam += np.log(d)
    return lam / N

# --- Divergence de trajectoires voisines
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
configs = [(3.2, 'periodique'), (3.9, 'chaotique')]

for col, (r, nom) in enumerate(configs):
    x1 = logistique(r, 0.500,  60)
    x2 = logistique(r, 0.5001, 60)
    delta = np.abs(x1 - x2)

    lam = lyapunov(r)
    axes[0, col].plot(x1, color='steelblue', lw=1.2,
                      label='$x_0 = 0{,}500$')
    axes[0, col].plot(x2, color='firebrick', lw=1.2, ls='--',
                      label='$x_0 = 0{,}5001$')
    axes[0, col].set_title(
        f'$r = {r}$ ({nom}), $\\lambda = {lam:.3f}$')
    axes[0, col].set_ylabel('$x_n$')
    axes[0, col].legend(fontsize=8)

    axes[1, col].semilogy(delta + 1e-16, color='purple', lw=1.2)
    axes[1, col].set_xlabel('Iteration $n$')
    axes[1, col].set_ylabel(r'$|\delta x_n|$ (log)')

plt.tight_layout()
plt.savefig(FIG / 'ergo_lyapunov.pdf', dpi=150)

# --- Spectre de Lyapunov
r_vals = np.linspace(2.5, 4.0, 700)
lam_vals = np.array([lyapunov(r) for r in r_vals])

plt.figure(figsize=(8, 4))
plt.plot(r_vals, lam_vals, 'k-', lw=0.7)
plt.axhline(0, color='firebrick', lw=1.5, ls='--')
plt.fill_between(r_vals, lam_vals, 0,
                 where=(lam_vals > 0),
                 color='orange', alpha=0.45,
                 label=r'chaos ($\lambda > 0$)')
plt.xlabel('Paramètre de contrôle $r$ (sans dimension)')
plt.ylabel(r'Exposant de Lyapunov $\lambda$ (sans unité)')
plt.legend()
plt.tight_layout()
plt.savefig(FIG / 'ergo_spectre.pdf', dpi=150)

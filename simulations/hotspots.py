"""L'entropie sous toutes ses formes -- chapitre « Entropie spatiale: villes, ségrégation et criminalité »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def lap(f, h):
    """Laplacien a 5 points, bords periodiques."""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1)
            + np.roll(f, -1, 1) - 4 * f) / h**2

def div_rho_grad(rho, g, h):
    """Divergence de rho*grad(g), flux centres sur les aretes."""
    out = np.zeros_like(rho)
    for ax in (0, 1):
        flux = 0.5*(rho + np.roll(rho, -1, ax)) * (np.roll(g, -1, ax) - g) / h
        out += (flux - np.roll(flux, 1, ax)) / h
    return out

def simulate_crime(N=50, T=47.0, h=0.2, dt=0.0005, eta=1.0, omega=1.0,
                   theta=1.0, Gamma=2.0, D_A=0.04, D_rho=1.0,
                   seed=0):
    rng = np.random.default_rng(seed)
    # etat homogene stationnaire, legerement bruite (omega = theta = 1)
    A = (eta + Gamma) * (1 + 0.01 * rng.standard_normal((N, N)))
    rho = np.full((N, N), Gamma / (eta + Gamma))
    H_history = []
    for n in range(int(T / dt)):
        crimes = rho * A
        if n % 200 == 0:                      # entropie spatiale
            p = crimes / crimes.sum()
            H_history.append((n * dt, -np.sum(p * np.log2(p))))
        dA = eta - omega*A + theta*crimes + D_A*lap(A, h)
        drho = D_rho*(lap(rho, h) - 2*div_rho_grad(rho, np.log(A), h)) \
               - crimes + Gamma
        A = np.maximum(A + dt*dA, 1e-3)
        rho = np.maximum(rho + dt*drho, 1e-9)
    return A, rho, np.array(H_history)

A, rho, H = simulate_crime()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(A, cmap='hot'); axes[0].set_title('Attractivité finale')
axes[1].imshow(rho, cmap='Blues'); axes[1].set_title('Densité délinquants')
axes[2].plot(H[:, 0], H[:, 1])
axes[2].set_xlabel('Temps'); axes[2].set_ylabel('bits')
axes[2].set_title('Entropie spatiale des crimes')
plt.tight_layout()
plt.savefig(FIG / 'hotspots_entropie.pdf')

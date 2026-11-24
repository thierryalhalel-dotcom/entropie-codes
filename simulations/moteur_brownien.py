"""L'entropie sous toutes ses formes -- chapitre « Moteurs moléculaires: quand la chaleur fait travailler »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


np.random.seed(42)
kBT = 1.0           # en unites adimensionnees
dt  = 0.01
N_pas = 5000
N_part = 200        # ensemble de particules

def potentiel_dent(x, amplitude=3.0, periode=2*np.pi):
    """Potentiel asymetrique en dents de scie."""
    x_mod = x % periode
    # Montee rapide (1/4 periode), descente lente (3/4 periode)
    pente_montee = amplitude / (periode * 0.25)
    pente_descente = -amplitude / (periode * 0.75)
    return np.where(x_mod < 0.25 * periode,
                    pente_montee * x_mod,
                    amplitude + pente_descente * (x_mod - 0.25*periode))

def force(x, amplitude=3.0, periode=2*np.pi):
    """Force = -dU/dx (differentiation numerique)."""
    dx = 1e-5
    return -(potentiel_dent(x+dx,amplitude,periode)
             - potentiel_dent(x-dx,amplitude,periode)) / (2*dx)

pos = np.zeros(N_part)
x_moy = [0.0]
T_clignote = 50   # periode de clignotement (pas)

for t in range(N_pas):
    # Potentiel actif ou eteint en alternance
    actif = (t % T_clignote) < (T_clignote // 2)
    F = force(pos) if actif else np.zeros(N_part)
    bruit = np.sqrt(2 * kBT * dt) * np.random.randn(N_part)
    pos += F * dt + bruit   # equation de Langevin
    x_moy.append(pos.mean())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x_moy, color='steelblue', lw=1)
axes[0].set_xlabel('Pas de temps (sans dim.)')
axes[0].set_ylabel('Position moyenne (sans dim.)')
axes[0].set_title('Transport dirige par le moteur brownien')

x_range = np.linspace(0, 4*np.pi, 300)
axes[1].plot(x_range, potentiel_dent(x_range),
             color='firebrick', lw=2)
axes[1].set_xlabel('Position (sans dim.)')
axes[1].set_ylabel('Potentiel $U(x)$ (unites $k_BT$)')
axes[1].set_title('Potentiel asymetrique (dents de scie)')
plt.tight_layout()
plt.savefig(FIG / 'moteur_brownien.pdf', dpi=150)

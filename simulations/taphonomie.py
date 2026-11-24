"""L'entropie sous toutes ses formes -- chapitre « Entropie et archéologie: le temps qui efface »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


# Demi-vies des matériaux (années)
materiaux = {
    'Os (sol neutre)': 5_000,
    'Bois (sol humide)': 1_000,
    'Céramique': 200_000,
    'Parchemin (sec)': 800,
}

t = np.linspace(0, 20_000, 500)
fig, ax = plt.subplots(figsize=(8, 5))
for mat, t12 in materiaux.items():
    lam = np.log(2) / t12
    I = np.exp(-lam * t)
    ax.plot(t, I * 100, label=mat)
ax.set_xlabel('Temps (années)')
ax.set_ylabel('Information conservée (%)')
ax.set_title('Taux de survie informationnelle (taphonomie)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / 'taphonomie_survie.pdf')

# Incertitude de datation 14C
A0 = 1.0
t_true = np.array([1000, 3000, 5000, 8000, 12000])
t12_C14 = 5730
lam_C14 = np.log(2) / t12_C14
A_measured = A0 * np.exp(-lam_C14 * t_true)
sigma_A = 0.01  # incertitude de mesure sur A (1%)
sigma_t = sigma_A / (lam_C14 * A_measured)
print("Âge (ans) | Activité | Incertitude (ans)")
for tt, A, st in zip(t_true, A_measured, sigma_t):
    print(f"  {tt:6d}   |  {A:.4f}  |  ±{st:.0f}")

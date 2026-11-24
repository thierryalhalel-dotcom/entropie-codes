"""L'entropie sous toutes ses formes -- chapitre « Entropie bureaucratique: la norme entre mesure et métaphore »."""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


# --- Croissance exponentielle du corpus ---
t = np.arange(0, 60)          # annees depuis 1970
lam = 0.025                    # taux de croissance annuel (2,5 %)
N0 = 10_000                    # articles en 1970
N = N0 * np.exp(lam * t)

# Jurisprudence generee (mu articles de jurisprudence par article par an)
mu = 0.20
J = (mu / lam) * N0 * (np.exp(lam * t) - 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.semilogy(1970 + t, N,   color='steelblue', lw=2, label='Articles de loi')
ax.semilogy(1970 + t, J,   color='firebrick', lw=2,
            ls='--', label='Jurisprudence cumulée')
ax.semilogy(1970 + t, N+J, color='forestgreen', lw=1.5,
            ls=':', label='Total')
ax.set_xlabel('Année')
ax.set_ylabel('Nombre (échelle log)')
ax.set_title(f'Croissance normative\n'
             r'$N(t)=N_0 e^{\lambda t}$, $\lambda=2{,}5\,\%$/an')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# --- Entropie lexicale simulee ---
# Texte type loi (3 mots tres frequents + vocabulaire technique rare)
# vs texte pedagogique (distribution Zipf plus douce)
np.random.seed(42)
V_loi  = 300    # vocabulaire
V_peda = 200

# Loi technique : 3 mots super-frequents + queue plate
freq_loi  = np.ones(V_loi)
freq_loi[:3] = 150
p_loi = freq_loi / freq_loi.sum()

# Texte pedagogique : distribution Zipf douce
r_peda = np.arange(1, V_peda + 1)
p_peda = (1.0 / r_peda)
p_peda /= p_peda.sum()

def H_bits(p):
    p = p[p > 1e-15]
    return -np.sum(p * np.log2(p))

ax = axes[1]
styles = [('Loi technique', p_loi, 'steelblue'),
          ('Texte pédagogique', p_peda, 'forestgreen')]
for nom, p, col in styles:
    H = H_bits(p)
    ax.plot(np.sort(p)[::-1], color=col, lw=1.8,
            label=f'{nom}  $H$={H:.1f} bits')

ax.set_xlabel('Rang du mot (par fréquence décroissante)')
ax.set_ylabel('Probabilité (sans dim.)')
ax.set_title('Distributions lexicales\n'
             r'$H = -\sum p_w \log_2 p_w$')
ax.legend(fontsize=9)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(FIG / 'bureaucratie_entropie.pdf', dpi=150)

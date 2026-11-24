"""L'entropie sous toutes ses formes -- chapitre « Entropie conditionnelle et théorème de Bayes »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


np.random.seed(42)

prevalence = 0.001
sensibilite = 0.99
specificite = 0.95

def H_bin(p):
    """Entropie binaire en bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def bayes_update(prior, resultat_positif, sens, spec):
    """Mise a jour bayesienne."""
    if resultat_positif:
        vrais = sens * prior
        faux_pos = (1 - spec) * (1 - prior)
        return vrais / (vrais + faux_pos)
    else:
        vrais_neg = spec * (1 - prior)
        faux_neg = (1 - sens) * prior
        return (1 - sens) * prior / ((1 - sens) * prior + vrais_neg)

n_tests = 8
est_malade = True

posterior = prevalence
entropies = [H_bin(posterior)]
posteriors = [posterior]

for i in range(n_tests):
    if est_malade:
        positif = np.random.random() < sensibilite
    else:
        positif = np.random.random() > specificite
    posterior = bayes_update(posterior, positif, sensibilite, specificite)
    posteriors.append(posterior)
    entropies.append(H_bin(posterior))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(range(n_tests + 1), posteriors, 'o-', color='steelblue', lw=1.5)
ax1.set_ylabel('$P$(malade)')
ax1.axhline(0.5, ls='--', color='gray', lw=0.8)
ax1.set_title('Mise à jour bayésienne séquentielle')

ax2.plot(range(n_tests + 1), entropies, 's-', color='firebrick', lw=1.5)
ax2.set_ylabel('$H$ (bits)')
ax2.set_xlabel('Nombre de tests')

plt.tight_layout()
plt.savefig(FIG / 'bayes_entropie.pdf', dpi=150)

"""L'entropie sous toutes ses formes -- chapitre « L'entropie relative: mesurer l'écart entre deux mondes »."""
import numpy as np
from collections import Counter

def distribution(texte, alphabet=None):
    """Distribution de probabilite des caracteres."""
    texte = texte.lower().replace(' ', '')
    cnt = Counter(texte)
    if alphabet is None:
        alphabet = sorted(cnt.keys())
    total = sum(cnt.values())
    n = len(alphabet)   # lissage de Laplace : aucune probabilite nulle
    return np.array([(cnt.get(c, 0) + 1) / (total + n)
                     for c in alphabet]), alphabet

def kl_divergence(p, q, base=2):
    """D_KL(P||Q) en bits (base=2) ou nats (base=e)."""
    eps = 1e-12
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    if base == 2:
        return np.sum(p * np.log2(p / q))
    return np.sum(p * np.log(p / q))

# Trois "sources"
fr  = "la thermodynamique est la science de la chaleur"
en  = "thermodynamics is the science of heat and energy"
ale = "xkqzmvbpwjrtlfhgdsnuoaeyixkqzmvbpwjrtlfhg"

# Alphabet commun
alphabet = sorted(set(fr.replace(' ','') +
                       en.replace(' ','') +
                       ale.replace(' ','')))

p_fr,  _ = distribution(fr,  alphabet)
p_en,  _ = distribution(en,  alphabet)
p_ale, _ = distribution(ale, alphabet)

print(f"D_KL(FR || EN)   = {kl_divergence(p_fr, p_en):.3f} bits")
print(f"D_KL(EN || FR)   = {kl_divergence(p_en, p_fr):.3f} bits")
print(f"D_KL(FR || alea) = {kl_divergence(p_fr, p_ale):.3f} bits")
print(f"D_KL(FR || FR)   = {kl_divergence(p_fr, p_fr):.6f} bits")

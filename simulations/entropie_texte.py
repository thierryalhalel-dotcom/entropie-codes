"""L'entropie sous toutes ses formes -- chapitre « Shannon: l'entropie de l'information »."""
import numpy as np
from collections import Counter

def H_shannon(texte):
    """Entropie de Shannon en bits par caractère."""
    texte = texte.lower().replace(' ', '')
    counts = np.array(list(Counter(texte).values()), dtype=float)
    p = counts / counts.sum()                       # probabilités
    return -np.sum(p * np.log2(p + 1e-15))          # +eps évite log(0)

exemples = {
    'Texte français':
        "la thermodynamique est la science de la chaleur",
    'Texte aléatoire':
        "xkqzmvbpwjrtlfhgdsnuoaeyixkqzmvbpwjrt",
    'Un seul caractère':
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
}

print(f"{'Source':<22} | {'H (bits/car.)'}")
print("-" * 38)
for nom, txt in exemples.items():
    H = H_shannon(txt)
    print(f"{nom:<22} | {H:.3f} bits/car.")

# Maximum théorique pour l'alphabet français (26 lettres)
H_max = np.log2(26)
print(f"\nMaximum théorique (26 lettres) : {H_max:.3f} bits/car.")

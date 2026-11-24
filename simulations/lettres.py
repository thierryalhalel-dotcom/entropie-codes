"""L'entropie sous toutes ses formes -- chapitre « Shannon ».

Entropie de la lettre suivante sachant les k precedentes, mesuree sur
Les Trois Mousquetaires (27 symboles : 26 lettres sans accent et l'espace).
Le modele est appris sur 90 % du roman et teste sur les 10 % restants.
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG = RACINE / "figures"
FIG.mkdir(exist_ok=True)
t = (RACINE / "donnees/trois_mousquetaires.txt").read_text(encoding="utf8")
t = t[t.rindex("CHAPITRE PREMIER"):].lower().replace("œ", "oe")
t = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
t = re.sub(r"[^a-z]+", " ", t)
n = int(0.9 * len(t))
A, B = t[:n], t[n:]
KMAX = 8
comptes = [Counter(A[i:i + k] for i in range(len(A) - k + 1)) for k in range(KMAX + 2)]


def proba(ctx, c):
    """Melange des ordres 0 a k (interpolation de Witten-Bell simplifiee)."""
    p = 1 / 27
    for k in range(len(ctx) + 1):
        s = ctx[len(ctx) - k:]
        tot = comptes[k][s] if k else len(A)
        if tot == 0:
            break
        lam = tot / (tot + 5)
        p = lam * comptes[k + 1][s + c] / tot + (1 - lam) * p
    return p


H = [np.log2(27)]
pos = range(KMAX, len(B), 3)
for k in range(KMAX + 1):
    H.append(-np.mean([np.log2(proba(B[i - k:i], B[i])) for i in pos]))
    print(k, round(H[-1], 2))
plt.figure(figsize=(7.5, 4))
plt.plot(range(-1, KMAX + 1), H, "o-", color="firebrick")
plt.xticks(range(-1, KMAX + 1), ["aucune\nstatistique"] + list(range(KMAX + 1)))
plt.axhspan(0.6, 1.3, color="steelblue", alpha=0.2, label="lecteurs humains (Shannon 1951, anglais)")
plt.xlabel("Nombre de lettres précédentes connues")
plt.ylabel("Incertitude sur la lettre suivante (bits)")
plt.ylim(0, 5)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "lettres.pdf")

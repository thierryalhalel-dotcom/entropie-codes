"""L'entropie sous toutes ses formes -- chapitre « Entropie et musique »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
from collections import Counter

rng = np.random.default_rng(12)
# hauteurs seulement (sans rythme) ; g, a, b minuscules : octave inferieure
clair_a = "C C C D E D C E D D C".split()
clair_b = "D D D D a a D C b a g".split()
clair = clair_a * 2 + clair_b + clair_a
jacques = ("C D E C".split() * 2 + "E F G".split() * 2
           + "G A G F E C".split() * 2 + "C g C".split() * 2)
l1 = "E E F G G F E D C C D E E D D".split()
l2 = "E E F G G F E D C C D E D C C".split()
l3 = "D D E C D E F E C D E F E D C D g".split()
joie = l1 + l2 + l3 + l2
serie = "E F G C# F# D# G# D B C A A#".split() * 5   # serie de l'op. 25 (Schoenberg)
gamme = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
hasard = list(rng.choice(gamme, 60))


def H0(s):
    p = np.array(list(Counter(s).values())) / len(s)
    return -(p * np.log2(p)).sum()


def H1(s):
    """Entropie de la note suivante sachant la note courante."""
    paires = list(zip(s[:-1], s[1:]))
    return H0(paires) - H0(s[:-1])


noms = ["Frère\nJacques", "Au clair\nde la lune", "Ode\nà la joie",
        "série de\n12 sons", "notes\nau hasard"]
airs = [jacques, clair, joie, serie, hasard]
a, b = [H0(s) for s in airs], [H1(s) for s in airs]
for n, s, u, v in zip(noms, airs, a, b):
    print(repr(n), len(s), "notes", round(u, 2), round(v, 2))
X = np.arange(5)
plt.figure(figsize=(8.5, 4))
plt.bar(X - 0.2, a, 0.4, color="steelblue", label="$H$ : notes prises une à une")
plt.bar(X + 0.2, b, 0.4, color="firebrick", label="$H$ sachant la note précédente")
plt.axhline(np.log2(12), ls="--", color="gray", lw=1, label=r"$\log_2 12$")
plt.xticks(X, noms)
plt.ylabel("bits par note")
plt.legend(fontsize=8, loc="upper left")
plt.ylim(0, 4.6)
plt.tight_layout()
plt.savefig(FIG / "melodies.pdf")

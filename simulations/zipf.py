"""L'entropie sous toutes ses formes -- chapitre « Linguistique ».

Loi de Zipf sur le texte integral des Trois Mousquetaires (domaine public,
source : Project Gutenberg n. 13951) et sur un « singe dactylographe ».
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG = RACINE / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(8)

texte = (RACINE / "donnees/trois_mousquetaires.txt").read_text(encoding="utf8")
texte = texte[texte.rindex("CHAPITRE PREMIER"):].lower().replace("’", "'")
# les mots elides (l', d', qu') comptent comme des mots
mots = re.findall(r"[a-zàâäçéèêëîïôöùûüÿœæ]+'?", texte)
M = len(mots)
compte = Counter(mots)


def entropie(c):
    p = np.array(list(c.values()), float)
    p /= p.sum()
    return -(p * np.log2(p)).sum()


# singe : 26 lettres et espace equiprobables, meme nombre de mots
car = rng.integers(0, 27, 40 * M)
singe = "".join("abcdefghijklmnopqrstuvwxyz "[k] for k in car[:27 * M + 100]).split()[:M]
cs = Counter(singe)


def pente(c, r1, r2):
    f = np.array(sorted(c.values(), reverse=True), float)
    r = np.unique(np.logspace(np.log10(r1), np.log10(r2), 60).astype(int))
    return -np.polyfit(np.log(r), np.log(f[r - 1]), 1)[0]


print(f"{M} mots, {len(compte)} distincts, hapax {sum(v == 1 for v in compte.values())}")
print(f"H (mots) = {entropie(compte):.2f} bits ; alpha (10-3000) = {pente(compte, 10, 3000):.2f}")
print(f"singe : {len(cs)} distincts")
tot = [c for _, c in compte.most_common(8)]
for r, (m, k) in enumerate(compte.most_common(8), 1):
    print(r, m, k, round(k / M, 4), round(tot[0] / k, 2))
p = np.array(tot) / sum(tot)
print("somme f8", round(sum(tot) / M, 4), "p", p.round(3),
      "H8", round(-(p * np.log2(p)).sum(), 3),
      "alpha 1-8", round(np.log(tot[0] / tot[7]) / np.log(8), 3))
lettres = Counter(c for c in texte if c.isalpha())
print("H lettres", round(entropie(lettres), 2))
L = {m: len(m.strip("'")) for m in compte}
print("longueur moyenne des 100 premiers :",
      round(np.mean([L[m] for m, _ in compte.most_common(100)]), 2),
      "; rangs 1001-2000 :",
      round(np.mean([L[m] for m, _ in compte.most_common(2000)[1000:]]), 2))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for c, nom, coul in ((compte, "Les Trois Mousquetaires", "steelblue"),
                     (cs, "singe dactylographe", "gray")):
    f = np.array(sorted(c.values(), reverse=True)) / M
    ax[0].loglog(np.arange(1, len(f) + 1), f, ".", ms=3, color=coul, label=nom)
r = np.array([1, 2e4])
ax[0].loglog(r, 0.07 / r, "r--", lw=1, label="$1/r$")
ax[0].set_xlabel("Rang $r$")
ax[0].set_ylabel("Fréquence $f_r$")
ax[0].legend(fontsize=8)
rangs = np.array([1, 3, 10, 30, 100, 300, 1000, 3000])
freq = compte.most_common()
x, y = [], []
for a, b in zip(rangs[:-1], rangs[1:]):
    g = freq[a - 1:b - 1]
    x.append(np.mean([-np.log2(k / M) for _, k in g]))
    y.append(np.mean([L[m] for m, _ in g]))
ax[1].plot(x, y, "o-", color="firebrick")
ax[1].set_xlabel("Surprise moyenne $-\\log_2 f$ (bits)")
ax[1].set_ylabel("Longueur moyenne (lettres)")
ax[1].set_title("Les mots rares sont longs")
plt.tight_layout()
plt.savefig(FIG / "zipf.pdf")

"""L'entropie sous toutes ses formes -- chapitre « Linguistique ».

Le meme roman en francais (original) et en anglais (traduction) :
Les Trois Mousquetaires / The Three Musketeers (domaine public,
Project Gutenberg n. 13951 et n. 1257).
"""
import lzma
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG = RACINE / "figures"
FIG.mkdir(exist_ok=True)

debut = {"fr": "Le premier lundi du mois d’avril", "en": "On the first Monday of the month of April"}
fichier = {"fr": "trois_mousquetaires.txt", "en": "three_musketeers.txt"}
motif = {"fr": r"[a-zàâäçéèêëîïôöùûüÿœæ]+'?", "en": r"[a-zàâäçéèêëîïôöùûüÿœæ]+(?:'[a-z]+)?"}
nom = {"fr": "français", "en": "anglais"}


def entropie(c):
    p = np.array(list(c.values()), float)
    p /= p.sum()
    return -(p * np.log2(p)).sum()


def sans_accent(t):
    t = t.replace("œ", "oe").replace("æ", "ae")
    return "".join(c for c in unicodedata.normalize("NFD", t) if "a" <= c <= "z")


mots, compte, lettres = {}, {}, {}
for g in ("fr", "en"):
    t = (RACINE / "donnees" / fichier[g]).read_text(encoding="utf8")
    t = re.sub(r"\s+", " ", t)
    t = t[t.index(debut[g]):].lower().replace("’", "'")
    mots[g] = re.findall(motif[g], t)
    compte[g] = Counter(mots[g])
    lettres[g] = Counter(sans_accent(t))
    c, M = compte[g], len(mots[g])
    f = np.array(sorted(c.values(), reverse=True), float)
    r = np.unique(np.logspace(1, np.log10(3000), 60).astype(int))
    alpha = -np.polyfit(np.log(r), np.log(f[r - 1]), 1)[0]
    octets = len(lzma.compress(t.encode("utf8"), preset=9))
    print(f"{g}: {len(t)} caracteres, {M} mots, {len(c)} formes, "
          f"hapax {sum(v == 1 for v in c.values())}, H mots {entropie(c):.2f}, "
          f"alpha {alpha:.2f}, H lettres {entropie(lettres[g]):.3f}, "
          f"longueur {np.mean([len(m.strip(chr(39))) for m in mots[g]]):.2f}, "
          f"lzma {octets} octets = {8*octets/len(t):.2f} bits/car, {8*octets/M:.1f} bits/mot")
    for k, (m, v) in enumerate(c.most_common(8), 1):
        print("  ", k, m, v, f"{100*v/M:.2f} %", round(f[0] / v, 2))
    print("   part des 10 premiers", round(f[:10].sum() / M, 3), " 100:", round(f[:100].sum() / M, 3))

alph = "abcdefghijklmnopqrstuvwxyz"
p = {g: np.array([lettres[g][a] for a in alph], float) for g in ("fr", "en")}
p = {g: v / v.sum() for g, v in p.items()}
kl = lambda a, b: float((a * np.log2(a / b)).sum())
print(f"D(FR||EN) = {kl(p['fr'], p['en']):.3f}, D(EN||FR) = {kl(p['en'], p['fr']):.3f}, "
      f"D(FR||unif) = {kl(p['fr'], np.full(26, 1/26)):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for g, coul in (("fr", "steelblue"), ("en", "firebrick")):
    f = np.array(sorted(compte[g].values(), reverse=True)) / len(mots[g])
    ax[0].loglog(np.arange(1, len(f) + 1), f, ".", ms=3, color=coul, label=nom[g])
r = np.array([1, 2e4])
ax[0].loglog(r, 0.07 / r, "k--", lw=1, label="$1/r$")
for (m, v), (m2, v2), k in zip(compte["fr"].most_common(3), compte["en"].most_common(3), (1, 2, 3)):
    ax[0].annotate(m2, (k, v2 / len(mots["en"])), xytext=(4, 4), textcoords="offset points",
                   fontsize=8, color="firebrick")
    ax[0].annotate(m, (k, v / len(mots["fr"])), xytext=(-4, -11), textcoords="offset points",
                   fontsize=8, color="steelblue")
ax[0].set_xlabel("Rang $r$")
ax[0].set_ylabel("Fréquence $f_r$")
ax[0].legend(fontsize=8)
x = np.arange(26)
ax[1].bar(x - 0.2, 100 * p["fr"], 0.4, color="steelblue", label="français")
ax[1].bar(x + 0.2, 100 * p["en"], 0.4, color="firebrick", label="anglais")
ax[1].set_xticks(x)
ax[1].set_xticklabels(alph)
ax[1].set_ylabel("Fréquence des lettres (%)")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "zipf_bilingue.pdf")

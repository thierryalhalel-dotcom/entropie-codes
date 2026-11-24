"""L'entropie sous toutes ses formes -- chapitre « Entropie et musique ».

Donnees REELLES : partitions numeriques au format Humdrum **kern (C. S. Sapp,
https://github.com/craigsapp, licence CC BY-NC-SA 4.0). Les partitions ne sont
PAS redistribuees ici : elles sont telechargees (git) dans donnees/externe/kern.
Seules des statistiques en sont tirees.
"""
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG, EXT = RACINE / "figures", RACINE / "donnees/externe/kern"
FIG.mkdir(exist_ok=True)
CORPUS = {"Bach\n(chorals)": "bach-370-chorales", "Mozart\n(sonates)": "mozart-piano-sonatas",
          "Beethoven\n(sonates)": "beethoven-piano-sonatas", "Joplin\n(ragtimes)": "joplin"}
PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def hauteur(tok):
    """Classe de hauteur de la note la plus aigue du jeton, ou None."""
    best = None
    for n in tok.split():
        if "q" in n or "r" in n or "_" in n or "]" in n:
            continue                          # ornements, silences, liaisons
        m = re.search(r"([a-g]+|[A-G]+)([#-]*)", n)
        if m:
            l = m[1]
            o = len(l) - 1 if l.islower() else -len(l)
            alt = m[2].count("#") - m[2].count("-")
            h = 12 * o + PC[l[0].lower()] + alt
            best = h if best is None else max(best, h)
    return best


def tonique(t):
    m = re.fullmatch(r"\*([A-Ga-g])([#-]?):.*", t)
    if m:
        return (PC[m[1].lower()] + (m[2] == "#") - (m[2] == "-")) % 12, m[1].isupper()


def lire(f):
    """Suites (degre chromatique, mode majeur) par voix."""
    voix, types, suites, ton = [], None, {}, None
    nid = 0
    for lg in f.read_text(encoding="latin1").splitlines():
        if not lg or lg.startswith("!"):
            continue
        T = lg.split("\t")
        if lg.startswith("**"):
            voix = [(k, x == "**kern") for k, x in enumerate(T)]
            nid = len(T)
            continue
        if len(T) != len(voix):
            continue
        if lg.startswith("*"):
            for x in T:
                ton = tonique(x) or ton
            nv, i = [], 0
            while i < len(T):
                if T[i] == "*^":
                    nv += [voix[i], (nid, voix[i][1])]
                    nid += 1
                elif T[i] == "*v":
                    nv.append(voix[i])
                    while i + 1 < len(T) and T[i + 1] == "*v":
                        i += 1
                elif T[i] != "*-":
                    nv.append(voix[i])
                i += 1
            voix = nv
            continue
        if lg.startswith("=") or ton is None:
            continue
        for (v, k), tok in zip(voix, T):
            if k and tok != ".":
                h = hauteur(tok)
                if h is not None:
                    suites.setdefault(v, []).append(((h - ton[0]) % 12, ton[1]))
    return list(suites.values())


def H(c):
    p = np.array(list(c.values()), float)
    p /= p.sum()
    return -(p * np.log2(p)).sum()


res = {}
for nom, d in CORPUS.items():
    if not (EXT / d).exists():
        EXT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/craigsapp/{d}", str(EXT / d)], check=True)
    uni, av, bi, n = Counter(), Counter(), Counter(), 0
    fs = sorted((EXT / d).rglob("*.krn"))
    for f in fs:
        for s in lire(f):
            s = [x for x, maj in s if maj]        # mode majeur seulement
            uni.update(s)
            av.update(s[:-1])
            bi.update(zip(s[:-1], s[1:]))
    n = sum(uni.values())
    prof = np.array([uni[k] for k in range(12)]) / n
    res[nom] = (H(uni), H(bi) - H(av), n, prof)
    print(nom.replace("\n", " "), len(fs), "fichiers,", n, "notes (majeur) H0", round(res[nom][0], 2),
          "H1", round(res[nom][1], 2), "chromatique", round(prof[[1, 3, 6, 8, 10]].sum(), 3), prof.round(3))
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
X = np.arange(len(res))
ax[0].bar(X - 0.2, [v[0] for v in res.values()], 0.4, color="steelblue", label="$H$ : note seule")
ax[0].bar(X + 0.2, [v[1] for v in res.values()], 0.4, color="firebrick", label="$H$ sachant la précédente")
ax[0].axhline(np.log2(12), ls="--", color="gray", lw=1, label=r"$\log_2 12$")
ax[0].axhline(np.log2(7), ls=":", color="gray", lw=1, label=r"$\log_2 7$")
ax[0].set_xticks(X)
ax[0].set_xticklabels(res)
ax[0].set_ylim(0, 4.8)
ax[0].set_ylabel("bits par note")
ax[0].legend(fontsize=8, ncol=2, loc="upper center")
for (nom, v), c in zip(res.items(), ("k", "steelblue", "firebrick", "orange")):
    ax[1].plot(range(12), 100 * v[3], "o-", ms=4, color=c, label=nom.split("\n")[0])
ax[1].set_xticks(range(12))
ax[1].set_xticklabels(["1", "", "2", "", "3", "4", "", "5", "", "6", "", "7"])
ax[1].set_xlabel("Degré de la gamme majeure (sans numéro : notes altérées)")
ax[1].set_ylabel("Fréquence (%)")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "partitions.pdf")

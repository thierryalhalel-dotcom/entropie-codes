"""L'entropie sous toutes ses formes -- chapitre « Entropie spatiale ».

Donnees REELLES : populations municipales, https://geo.api.gouv.fr
(Etat francais, d'apres l'INSEE ; Licence Ouverte). Telecharge au besoin.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG, EXT = RACINE / "figures", RACINE / "donnees/externe"
FIG.mkdir(exist_ok=True)
EXT.mkdir(parents=True, exist_ok=True)
F = EXT / "communes.json"
if not F.exists():
    urllib.request.urlretrieve("https://geo.api.gouv.fr/communes?fields=nom,code,"
                               "population,surface,codeDepartement", F)
C = [c for c in json.loads(F.read_text(encoding="utf8"))
     if c.get("population") and len(c.get("codeDepartement", "")) == 2]
C.sort(key=lambda c: -c["population"])
p = np.array([c["population"] for c in C], float)
print(len(C), "communes (metropole)", int(p.sum()), "habitants")
for k, c in enumerate(C[:8], 1):
    print(k, c["nom"], c["population"], round(p[0] / c["population"], 2))


def H(x):
    x = np.asarray(x, float)
    x = x[x > 0] / x.sum()
    return -(x * np.log2(x)).sum()


r = np.arange(1, len(p) + 1)
q = (r >= 2) & (r <= 1000)
alpha = -np.polyfit(np.log(r[q]), np.log(p[q]), 1)[0]
print("H nationale", round(H(p), 2), "max", round(np.log2(len(p)), 2),
      "N_eff", round(2**H(p)), "alpha 2-1000", round(alpha, 2))
cum = np.cumsum(p) / p.sum()
print("communes portant la moitie de la population :", int((cum < 0.5).sum()) + 1)
dep = {}
for c in C:
    dep.setdefault(c["codeDepartement"], []).append(c)
T = []
for d, L in dep.items():
    if len(L) < 20:
        continue
    x = [c["population"] for c in L]
    T.append((H(x) / np.log2(len(x)), d, len(x), H(x), 2**H(x), max(L, key=lambda c: c["population"])["nom"],
              max(x) / sum(x)))
T.sort()
for l in T[:5] + T[-5:]:
    print(l[1], l[5], "N", l[2], "H", round(l[3], 2), "Neff", round(l[4]), "eff", round(l[0], 3), "part", round(l[6], 3))
for d in ("31", "13", "59", "69"):
    print([(l[1], l[5], l[2], round(l[3], 2), round(l[4]), round(l[0], 3), round(l[6], 2)) for l in T if l[1] == d])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(r, p, ".", ms=2, color="steelblue")
ax[0].loglog(r[:3000], p[0] / r[:3000], "k--", lw=1, label="$P_1/r$ (Zipf)")
for k, dy in ((0, 1.15), (1, 0.55), (3, 1.35)):
    ax[0].annotate(C[k]["nom"], (k + 1, p[k]), xytext=((k + 1) * 1.15, p[k] * dy), fontsize=8)
ax[0].set_xlabel("Rang")
ax[0].set_ylabel("Population")
ax[0].legend()
ax[1].scatter([l[6] * 100 for l in T], [l[0] for l in T], s=14, color="firebrick")
for l in T:
    if l in T[:3] + T[-2:] or l[1] in ("31", "59"):
        ax[1].annotate(l[1], (l[6] * 100, l[0]), xytext=(3, 3), textcoords="offset points", fontsize=8)
ax[1].set_xlabel("Part de la commune la plus peuplée (%)")
ax[1].set_ylabel(r"$H/\log_2 N$ (départements)")
plt.tight_layout()
plt.savefig(FIG / "villes_reelles.pdf", dpi=200)

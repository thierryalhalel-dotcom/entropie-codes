"""L'entropie sous toutes ses formes -- chapitre « Entropie en médecine ».

Donnees REELLES : intervalles RR de PhysioNet (licence ODC-By 1.0) :
nsr2db (54 sujets sains), chf2db (29 insuffisances cardiaques),
afdb (25 enregistrements avec fibrillation auriculaire). Telecharge au besoin.
Usage pedagogique uniquement : ce n'est pas un outil de diagnostic.
"""
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parent.parent
FIG, EXT = RACINE / "figures", RACINE / "donnees/externe"
FIG.mkdir(exist_ok=True)
URL = "https://physionet.org/files/{}/1.0.0/{}"


def fichier(base, nom):
    f = EXT / base / nom
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(base, nom), f)
    return f


def annotations(f):
    """Lecture du format MIT : liste (echantillon, code, texte auxiliaire)."""
    b = f.read_bytes()
    i, t, out = 0, 0, []
    while i + 1 < len(b):
        v = b[i] | b[i + 1] << 8
        i += 2
        code, d = v >> 10, v & 0x3FF
        if v == 0:
            break
        if code == 59:                       # SKIP : intervalle sur 4 octets
            d32 = (b[i] | b[i + 1] << 8) << 16 | (b[i + 2] | b[i + 3] << 8)
            t += d32 - (1 << 32) if d32 >> 31 else d32
            i += 4
        elif code == 63:                     # AUX : texte attache au precedent
            out[-1] = (out[-1][0], out[-1][1], b[i:i + d].decode("latin1").strip("\0"))
            i += d + d % 2
        elif code >= 60:
            pass
        else:
            t += d
            out.append((t, code, ""))
    return out


def sampen(x, m=2, r=0.2):
    r, n = r * x.std(), len(x)
    E = np.array([x[i:n - m + i] for i in range(m + 1)]).T
    def paires(k):
        d = np.abs(E[:, None, :k] - E[None, :, :k]).max(2)
        return ((d < r).sum() - len(E)) / 2
    a, b = paires(m + 1), paires(m)
    return -np.log(a / b) if a > 0 else np.nan


W = 1000                                     # battements par fenetre


def fenetres(rr, ok, nmax=8):
    """Fenetres de W intervalles valides, reparties sur l'enregistrement."""
    x = rr[ok & (rr > 300) & (rr < 2000)]
    n = len(x) // W
    return [x[k * W:(k + 1) * W] for k in np.unique(np.linspace(0, n - 1, min(n, nmax)).astype(int))]


def mesure(L):
    L = [w for w in L if w.std() > 0]
    return (np.median([sampen(w) for w in L]), np.median([w.mean() for w in L]),
            np.median([w.std() for w in L]))


G = {"sains": [], "insuffisance cardiaque": [], "fibrillation": []}
ex = {}
for nom, base, n, pre in (("sains", "nsr2db", 54, "nsr0"), ("insuffisance cardiaque", "chf2db", 29, "chf2")):
    for k in range(1, n + 1):
        a = annotations(fichier(base, f"{pre}{k:02d}.ecg"))
        a = [z for z in a if 1 <= z[1] <= 13 or z[1] in (25, 30, 34, 35, 38)]
        t = np.array([z[0] for z in a])
        N = np.array([z[1] == 1 for z in a])
        L = fenetres(np.diff(t) * 1000 / 128, N[1:] & N[:-1])
        G[nom].append(mesure(L))
        ex.setdefault(nom, L[len(L) // 2])
for rec in (RACINE / "donnees/afdb_RECORDS.txt").read_text().split():
    a = [z for z in annotations(fichier("afdb", rec + ".qrs"))]
    ryt = [z for z in annotations(fichier("afdb", rec + ".atr")) if z[2].startswith("(")]
    t = np.array([z[0] for z in a])
    j = np.searchsorted([z[0] for z in ryt], t, side="right") - 1
    af = np.array([k >= 0 and ryt[k][2] == "(AFIB" for k in j])
    L = fenetres(np.diff(t) * 4.0, af[1:] & af[:-1])
    if L:
        G["fibrillation"].append(mesure(L))
        ex.setdefault("fibrillation", L[0])
for k, v in G.items():
    v = np.array(v)
    print(k, "n =", len(v), "| SampEn mediane", np.median(v[:, 0]).round(2), "quartiles",
          np.percentile(v[:, 0], [25, 75]).round(2), "| RR", np.median(v[:, 1]).round(0),
          "| ecart-type", np.median(v[:, 2]).round(0))
s, c = (np.array(G[k])[:, 0] for k in ("sains", "insuffisance cardiaque"))
print("P(SampEn sain > SampEn IC) =", round(np.mean(s[:, None] > c[None]), 2))
s, c = (np.array(G[k])[:, 2] for k in ("sains", "insuffisance cardiaque"))
print("P(sigma sain > sigma IC) =", round(np.mean(s[:, None] > c[None]), 2))
fig, ax = plt.subplots(1, 4, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
for a, (k, coul) in zip(ax, (("sains", "forestgreen"), ("insuffisance cardiaque", "steelblue"),
                             ("fibrillation", "firebrick"))):
    a.plot(ex[k][:300], lw=0.6, color=coul)
    a.set_title(k + " (un exemple)", fontsize=9)
    a.set_ylim(300, 1500)
    a.set_xlabel("Battement n°")
ax[0].set_ylabel("Intervalle RR (ms)")
ax[3].boxplot([np.array(v)[:, 0] for v in G.values()], widths=0.6)
ax[3].set_xticklabels([f"{k.replace(' ', chr(10))}\n(n = {len(v)})" for k, v in G.items()], fontsize=8)
ax[3].set_ylabel("SampEn ($m=2$, $r=0{,}2\\,\\sigma$)")
plt.tight_layout()
plt.savefig(FIG / "rr_reels.pdf")

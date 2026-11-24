"""L'entropie sous toutes ses formes -- chapitre « Bayes » : 1000 patients."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
n, prev, sens, spec = 1000, 0.001, 0.99, 0.95
malades = round(n * prev)
vrais = round(malades * sens)
faux = round((n - malades) * (1 - spec))
etat = np.zeros(n, int)
etat[:faux] = 1
etat[faux:faux + vrais] = 2
print(malades, vrais, faux, round(vrais / (vrais + faux), 3))
y, x = np.divmod(np.arange(n), 50)
fig, ax = plt.subplots(figsize=(10, 4.4))
for k, (c, l) in enumerate((("lightgray", f"sain, test négatif ({n - faux - vrais})"),
                            ("orange", f"sain, test POSITIF ({faux})"),
                            ("red", f"malade, test positif ({vrais})"))):
    q = etat == k
    ax.scatter(x[q], -y[q], s=95 if k == 2 else 32, marker="*" if k == 2 else "o",
               color=c, edgecolor="k" if k == 2 else "none", label=l, zorder=k)
ax.axis("off")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)
plt.tight_layout()
plt.savefig(FIG / "patients.pdf")

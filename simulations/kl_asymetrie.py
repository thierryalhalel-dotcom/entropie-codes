"""L'entropie sous toutes ses formes -- chapitre « Entropie relative »."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
p = np.linspace(0.005, 0.995, 400)
kl = lambda a, b: a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
P, Q = np.array([0.9, 0.1]), np.array([0.5, 0.5])
ax[0].bar([0, 1], P, 0.35, color="steelblue", label="$P$ (réalité)")
ax[0].bar([0.37, 1.37], Q, 0.35, color="lightgray", label="$Q$ (modèle)")
ax[0].set_xticks([0.18, 1.18])
ax[0].set_xticklabels(["symbole 1", "symbole 2"])
ax[0].set_ylabel("Probabilité")
ax[0].legend()
ax[1].plot(p, kl(p, 0.5), color="firebrick", label=r"$D(P\|Q)$ : code uniforme, réalité $p$")
ax[1].plot(p, kl(0.5, p), color="steelblue", label=r"$D(Q\|P)$ : code pour $p$, réalité uniforme")
ax[1].plot([0.9, 0.9], [kl(0.9, 0.5), kl(0.5, 0.9)], "ko")
ax[1].set_ylim(0, 2.5)
ax[1].set_xlabel("$p$ (avec $P = (p, 1-p)$, $Q$ uniforme)")
ax[1].set_ylabel("Surcoût (bits par symbole)")
ax[1].legend(fontsize=8, loc="upper center")
print(round(kl(0.9, 0.5), 3), round(kl(0.5, 0.9), 3))
plt.tight_layout()
plt.savefig(FIG / "kl_asymetrie.pdf")

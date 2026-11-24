"""L'entropie sous toutes ses formes -- chapitre « Le rasoir d'Occam thermodynamique: pourquoi les lois sont simples »."""
import zlib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def complexite_approx(data_bytes):
    """Approximation de K(x) par compression zlib (en bits)."""
    return len(zlib.compress(data_bytes, level=9)) * 8

n = 10000  # longueur des sequences

resultats = {}

# 1. Suite reguliere : 0101...
reg = bytes([0, 1] * (n // 2))
resultats['Reguliere\n(01 repete)'] = complexite_approx(reg)

# 2. Suite aleatoire uniforme
ale = np.random.randint(0, 256, n, dtype=np.uint8).tobytes()
resultats['Aleatoire\nuniforme'] = complexite_approx(ale)

# 3. Bruit gaussien discret (distribution non uniforme)
gauss = np.clip(
    np.random.normal(128, 20, n).astype(int), 0, 255
).astype(np.uint8).tobytes()
resultats['Bruit\ngaussien'] = complexite_approx(gauss)

# 4. Pi en base 256
def pi_bytes(n):
    """Premiers n octets de pi (via mpmath)."""
    try:
        from mpmath import mp, pi
        mp.prec = 8 * n + 64
        entier = int(mp.floor(pi * mp.mpf(256)**n))
        return entier.to_bytes(n + 1, 'big')[-n:]
    except ImportError:
        return np.random.randint(0, 256, n, dtype=np.uint8).tobytes()

pi_data = pi_bytes(n)
resultats['Decimales\nde pi'] = complexite_approx(pi_data)

labels = list(resultats.keys())
vals = [v / n for v in resultats.values()]  # bits par octet

plt.figure(figsize=(7, 4))
plt.bar(labels, vals, color=['steelblue','firebrick','forestgreen','orange'],
        alpha=0.85)
plt.axhline(8, ls='--', color='gray', lw=1, label='Maximum = 8 bits/octet')
plt.ylabel('Complexite $K(x)/n$ (bits par octet)')
plt.title('Approximation de la complexite de Kolmogorov\npar compression zlib')
plt.legend()
plt.tight_layout()
plt.savefig(FIG / 'kolmogorov_compression.pdf', dpi=150)

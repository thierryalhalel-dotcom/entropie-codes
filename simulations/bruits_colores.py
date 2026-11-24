"""L'entropie sous toutes ses formes -- chapitre « Les couleurs du bruit: entropie spectrale et théorie du signal »."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


N, fe = 8192, 1000.0          # points, fréquence d'échantillonnage (Hz)
f = np.fft.rfftfreq(N, 1/fe)  # fréquences positives
f_pos = f[1:]                  # éviter f=0 pour la division

def make_noise(beta, N, fe):
    """Génère un bruit coloré d'exposant beta via filtrage spectral."""
    white = np.fft.rfft(np.random.randn(N))
    filter_ = np.concatenate([[1.0],
                               (f_pos / f_pos[0]) ** (-beta / 2)])
    return np.fft.irfft(white * filter_, n=N)

bruits = {
    'Blanc ($\\beta=0$)'   : 0,
    'Rose ($\\beta=1$)'    : 1,
    'Brun ($\\beta=2$)'    : 2,
    'Bleu ($\\beta=-1$)'   : -1,
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for nom, beta in bruits.items():
    x = make_noise(beta, N, fe)
    S = np.abs(np.fft.rfft(x))**2 / N
    axes[0].plot(f_pos, 10*np.log10(S[1:]+1e-30), label=nom, alpha=0.8)
    axes[1].plot(np.arange(500) / fe, x[:500] / x[:500].std() - 6 * len(axes[1].lines),
                 label=nom, lw=0.8, alpha=0.9)

axes[0].set_xlabel('Fréquence (Hz)')
axes[0].set_ylabel('DSP (dB)')
axes[0].set_xscale('log'); axes[0].legend(fontsize=8)
axes[0].set_title('Densités spectrales de puissance')

axes[1].set_xlabel('Temps (s)')
axes[1].set_ylabel('Amplitude normalisée (décalée)')
axes[1].set_yticks([])
axes[1].set_title('Signaux temporels (500 premiers points)')

plt.tight_layout()
plt.savefig(FIG / 'bruits_couleurs.pdf', dpi=150)

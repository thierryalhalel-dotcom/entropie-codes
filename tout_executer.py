"""Execute toutes les simulations, sans affichage."""
import os
import subprocess
import sys
from pathlib import Path

env = dict(os.environ, MPLBACKEND="Agg")
echecs = []
for script in sorted((Path(__file__).parent / "simulations").glob("*.py")):
    ok = subprocess.run([sys.executable, str(script)], env=env,
                        stdout=subprocess.DEVNULL).returncode == 0
    print("ok   " if ok else "ECHEC", script.name)
    if not ok:
        echecs.append(script.name)
sys.exit(bool(echecs))

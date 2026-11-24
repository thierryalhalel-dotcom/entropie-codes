"""Enregistre les mesures de l'entropimetre dans un fichier CSV.

Usage : python enregistrer.py /dev/ttyUSB0 mesures.csv     (pip install pyserial)
Commandes : taper m, a ou z puis Entree ; Ctrl-C pour terminer.
"""
import sys
import threading

import serial

port, fichier = sys.argv[1], sys.argv[2]
liaison = serial.Serial(port, 115200, timeout=1)


def clavier():
    for ligne in sys.stdin:
        liaison.write(ligne.strip().encode("ascii", "ignore"))


threading.Thread(target=clavier, daemon=True).start()
with open(fichier, "w") as f:
    try:
        while True:
            ligne = liaison.readline().decode("ascii", "ignore").strip()
            if ligne:
                print(ligne)
                if not ligne.startswith("#"):
                    f.write(ligne + "\n")
                    f.flush()
    except KeyboardInterrupt:
        liaison.write(b"a")     # on coupe toujours la chauffe

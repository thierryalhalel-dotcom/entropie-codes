# L'entropie sous toutes ses formes : codes Python

Dépôt : <https://github.com/thierryalhalel-dotcom/entropie-codes>

Codes des simulations du livre *L'entropie sous toutes ses formes* de Thierry Alhalel.

## Installation

```bash
pip install -r requirements.txt
python simulations/diffusion.py
```

Les figures sont enregistrées dans `figures/`. Pour tout exécuter (quelques minutes) :

```bash
python tout_executer.py
```

## Simulations

| Chapitre | Script |
|---|---|
| S ne fait que croître | [`diffusion.py`](simulations/diffusion.py) |
| S ne fait que croître | [`mosaique.py`](simulations/mosaique.py) (option `--gif`) |
| Boltzmann | [`pieces.py`](simulations/pieces.py) |
| La flèche du temps | [`cafe_esp32.py`](simulations/cafe_esp32.py) |
| Compression | [`compression.py`](simulations/compression.py) |
| Onsager | [`peltier.py`](simulations/peltier.py) |
| Transitions de phase | [`ising.py`](simulations/ising.py) |
| Trafic routier | [`trafic.py`](simulations/trafic.py) |
| IA générative | [`diffusion_ia.py`](simulations/diffusion_ia.py) |
| Linguistique | [`zipf.py`](simulations/zipf.py) |
| Linguistique | [`zipf_bilingue.py`](simulations/zipf_bilingue.py) |
| Potentiels thermodynamiques | [`haber.py`](simulations/haber.py) |
| Ensembles statistiques | [`deux_niveaux.py`](simulations/deux_niveaux.py) |
| Divergence KL | [`kl_asymetrie.py`](simulations/kl_asymetrie.py) |
| Entropie différentielle | [`quantification.py`](simulations/quantification.py) |
| Évolution | [`derive_genetique.py`](simulations/derive_genetique.py) |
| Holographie | [`loi_aire.py`](simulations/loi_aire.py) |
| Économie | [`echanges.py`](simulations/echanges.py) |
| Musique | [`melodies.py`](simulations/melodies.py) |
| Civilisations | [`tainter.py`](simulations/tainter.py) |
| Médecine | [`sampen.py`](simulations/sampen.py) |
| Shannon | [`lettres.py`](simulations/lettres.py) |
| Bayes | [`patients.py`](simulations/patients.py) |
| L'entropimètre : mesurer l'entropie | [`entropimetre.py`](simulations/entropimetre.py) |
| Boltzmann: le désordre se compte | [`boite_boltzmann.py`](simulations/boite_boltzmann.py) |
| Shannon: l'entropie de l'information | [`entropie_texte.py`](simulations/entropie_texte.py) |
| Entropie conditionnelle et théorème de Bayes | [`bayes_sequentiel.py`](simulations/bayes_sequentiel.py) |
| Les couleurs du bruit: entropie spectrale et théorie du signal | [`bruits_colores.py`](simulations/bruits_colores.py) |
| L'entropie dans la cellule: repliement, ADN et évolution | [`repliement_hp.py`](simulations/repliement_hp.py) |
| Percolation: l'ordre qui émerge du hasard | [`percolation_site.py`](simulations/percolation_site.py) |
| Percolation: l'ordre qui émerge du hasard | [`percolation_lien.py`](simulations/percolation_lien.py) |
| Percolation: l'ordre qui émerge du hasard | [`feu_de_foret.py`](simulations/feu_de_foret.py) |
| Entropie et fractales | [`fractales.py`](simulations/fractales.py) |
| Réseaux et graphes aléatoires: l'entropie des connexions | [`graphes_aleatoires.py`](simulations/graphes_aleatoires.py) |
| Le démon de Maxwell | [`reversibilite.py`](simulations/reversibilite.py) |
| La flèche du temps | [`refroidissement_cafe.py`](simulations/refroidissement_cafe.py) |
| Entropie et inégalité | [`courbes_lorenz.py`](simulations/courbes_lorenz.py) |
| Entropie spatiale: villes, ségrégation et criminalité | [`hotspots.py`](simulations/hotspots.py) |
| Gastronomie: thermodynamique dans l'assiette | [`gastronomie.py`](simulations/gastronomie.py) |
| Entropie bureaucratique: la norme entre mesure et métaphore | [`bureaucratie.py`](simulations/bureaucratie.py) |
| Entropie et jeux: les échecs, le poker, le go | [`echecs_bluff.py`](simulations/echecs_bluff.py) |
| Entropie et politique: la polarisation comme transition de phase | [`deffuant.py`](simulations/deffuant.py) |
| Entropie et archéologie: le temps qui efface | [`taphonomie.py`](simulations/taphonomie.py) |
| Entropie et arts visuels: la complexité du beau | [`entropie_image.py`](simulations/entropie_image.py) |
| Entropie et propagation virale: quand l'information se répand | [`sir_reseau.py`](simulations/sir_reseau.py) |
| Entropie et ergodicité | [`carte_logistique.py`](simulations/carte_logistique.py) |
| L'entropie relative: mesurer l'écart entre deux mondes | [`divergence_kl.py`](simulations/divergence_kl.py) |
| Moteurs moléculaires: quand la chaleur fait travailler | [`moteur_brownien.py`](simulations/moteur_brownien.py) |
| L'entropie quantique: de Von Neumann aux qubits | [`entropie_qubit.py`](simulations/entropie_qubit.py) |
| Le rasoir d'Occam thermodynamique: pourquoi les lois sont simples | [`kolmogorov_zlib.py`](simulations/kolmogorov_zlib.py) |
| La mort thermique : le destin entropique de l'Univers | [`mort_thermique.py`](simulations/mort_thermique.py) |

## Entropimètre (ESP32)

`arduino/entropimetre/entropimetre.ino` : programme de l'ESP32, à téléverser avec l'IDE Arduino (non encore testé sur matériel).
`arduino/peltier/peltier.ino` : module Peltier (non encore testé).
`arduino/enregistrer.py` : enregistrement CSV (`pip install pyserial`).
`donnees/` : mesures **simulées** (fichiers `*_SIMULE.csv`) et textes français et anglais des *Trois Mousquetaires* (domaine public).

## Figures du livre

`figures_livre/` contient les deux scripts qui produisent une partie des figures imprimées
(à lancer depuis ce dossier).

## Licence

Codes sous licence MIT (voir `LICENSE`). Cette licence ne couvre pas le texte du livre.

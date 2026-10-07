# zeta+ — la fonction zêta de Riemann sur trois algèbres

Visualise ζ sur une grille du plan, dans trois algèbres différentes, puis
détecte et classe ses minima locaux (zéros triviaux vs non triviaux).

> **Dépendance commune : ce projet nécessite le dépôt [`lib`](../lib).**
> Voir [Installation](#installation).

## Les trois algèbres

| Mode | Espace calculé | Image obtenue |
|---|---|---|
| `complex` | `ζ(x + i·y)` — algèbre usuelle | la droite critique, ses zéros, les pôles |
| `split` | ζ sur les split-complexes (`j² = +1`), via `ζ(x+y)` et `ζ(x−y)` | une structure en quadrillage de droites de pente 1 |
| `dual` | ζ sur les duals (`ε² = 0`) : donne `ζ(x)` **et** `ζ'(x)` en une passe | ζ et sa dérivée superposées |

## Ce que produit chaque mode

1. **Image colorée** — teinte = `arg(ζ)`, valeur = `log|ζ|` normalisé
   (percentiles 1 et 99).
2. **Carte des minima locaux** de `|ζ|`, séparés en :
   - **triviaux** (bleu) : sur la droite réelle, `x = −2, −4, …` (et leurs
     symétries selon le mode) ;
   - **non triviaux** (rouge) : les zéros de la droite critique
     `Re(s) = 1/2`.
3. **CSV** des zéros non triviaux avec `y ≥ 0` :
   `zeta_minima/nontrivial_{complex,split,dual}.csv` (colonnes `x,y`).

```
zeta_images/    images de la fonction
zeta_minima/    cartes de minima + CSV de zéros
```

## Installation

```bash
git clone https://github.com/<ton_compte>/lib.git
cd lib && pip install numpy numba
```

Puis, à côté du dépôt courant :

```bash
ln -s ../lib/lib.py .
```

Dépendances du projet :

```bash
pip install numpy mpmath scipy matplotlib tqdm
```

## Utilisation

```bash
python "zeta+.py"
```

Le script demande ensuite à l'interactive les modes à lancer :

```
=== Sélection des ensembles à calculer ===
Entrez une ou plusieurs options parmi : complex, split, dual
Votre sélection : complex dual
```

À la fin, les figures s'affichent (`plt.show()`).

## Réglages

Dans `config`, en tête de fichier :

```python
config = {
    'complex': {'range': (-20, 10, -30, 30), 'ppu': 10},
    'split':   {'range': (-20, 30, -15, 15),  'ppu': 10},
    'dual':    {'range': (-20, 10, -30, 30),  'ppu': 100},
}
```

`range` = `(x_min, x_max, y_min, y_max)`, `ppu` = points par unité.
Le coût est en `O(x_range·ppu × y_range·ppu)` appels à `mpmath`,
répartis sur tous les cœurs : **`split` à `ppu=100` est de loin le plus
long** (des minutes sur une machine modeste).

## Notes

- Les appels `mpmath.zeta` sont parallélisés via `multiprocessing.Pool`
  avec un générateur paresseux : aucune liste de coordonnées n'est
  construite en mémoire.
- En mode `dual`, `mpmath.diff` fournit `ζ'(x)` analytiquement ; la
  dépendance en `y` est linéaire (`ζ(x + y·ε) = ζ(x) + y·ζ'(x)·ε`), donc
  une seule colonne est calculée puis étalée.
- Les NaN (points non convergés par mpmath) sont tolérés : `log_magnitude`
  les convertit en `nanpercentile` puis les borne.

## Licence

Projet personnel — Paul Fournier.

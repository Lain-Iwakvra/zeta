"""zeta+.py — la fonction zêta de Riemann visualisée sur trois algèbres.

Calcule ζ sur une grille du plan et affiche, pour chaque algèbre :

  • **complex** — ζ : x + i·y  ↦  ζ(x + iy)                 (algèbre usuelle)
  • **split**   — ζ sur les nombres split-complexes (j² = +1),
                  à partir de la moyenne/différence ζ(x+y), ζ(x−y)
  • **dual**    — ζ sur les nombres duals (ε² = 0), ce qui donne
                  simultanément ζ(x) ET sa dérivée ζ'(x)

Pour chacun, le script produit :
  1. une image colorée (teinte = argument, valeur = magnitude en log) ;
  2. la carte des minima locaux de |ζ|, séparés en triviaux (sur la droite
     réelle, x = −2, −4, …) et non triviaux (les zéros de la droite
     critique) ;
  3. un CSV des zéros non triviaux (`zeta_minima/nontrivial_*.csv`).

Sorties :
    zeta_images/   images de la fonction
    zeta_minima/   cartes de minima + CSV de zéros

Dépendances : numpy, mpmath, scipy, matplotlib, tqdm, **lib**
    (fournit `Dual`, `SplitComplex`, `hsv_color`, `log_magnitude`)

Usage :
    python "zeta+.py"
    # demande ensuite à l'interactive une ou plusieurs valeurs parmi :
    #     complex   split   dual
"""

import os
import time
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from lib import Dual, SplitComplex, hsv_color, log_magnitude

# Plages de calcul par algèbre : (x_min, x_max, y_min, y_max) et points par
# unité. Attention, le coût est en O((x_range·ppu)·(y_range·ppu)) appels
# mpmath, répartis sur tous les cœurs : 'split' à ppu=100 est le plus long.
config = {
    'complex': {'range': (-20, 10, -30, 30), 'ppu': 10},
    'split': {'range': (-20, 30, -15, 15), 'ppu': 10},
    'dual': {'range': (-20, 10, -30, 30), 'ppu': 100}
}

os.makedirs("zeta_images", exist_ok=True)
os.makedirs("zeta_minima", exist_ok=True)


def create_grid(x0, x1, y0, y1, ppu):
    """Grille de points (X, Y) plus les vecteurs d'abscisses/ordonnées."""
    x = np.linspace(x0, x1, int((x1 - x0) * ppu))
    y = np.linspace(y0, y1, int((y1 - y0) * ppu))
    return np.meshgrid(x, y, indexing='xy'), x, y


def hsv_to_rgb(h, v):
    """Coloration : teinte h, valeur v, saturation maximale."""
    return hsv_color(h, v)


def compute_log_magnitude(z):
    """Magnitude en log10, normalisée entre les percentiles 1 et 99."""
    return log_magnitude(z)



def find_local_minima(data, size=3, border=1):
    """Minima locaux de `data` par filtre de taille `size` (mode réflexion).

    La bordure `border` lignes/colonnes est écartée : les minima d'angle
    sont des artefacts du traitement aux limites.
    """
    min_filt = minimum_filter(data, size=size, mode='reflect')
    minima = (data == min_filt)
    if border:
        minima[[*range(border), *range(-border, 0)], :] = False
        minima[:, [*range(border), *range(-border, 0)]] = False
    return minima


def plot_trivial_lines_split(x_min, x_max, y_min, y_max):
    """Points triviaux attendus en algèbre split.

    Dans cette algèbre, les « zéros triviaux » se répartissent sur une
    famille de droites de pente 1 (ξ = −(k+k₂), η = ξ + 2k), pas sur des
    points isolés : on les génère ici pour pouvoir les soustraire.
    """
    k_max = max(int(np.ceil((y_max - x_min) / 2)), int(np.ceil(-(y_min + x_min) / 2))) + 2
    x, y = [], []
    for k in range(1, k_max + 1):
        for k2 in range(1, k_max + 1):
            xi = -(k + k2)
            yi = xi + 2 * k
            if x_min <= xi <= x_max and y_min <= yi <= y_max:
                x.append(xi)
                y.append(yi)
    return np.array(x), np.array(y)


def classify_minima_split(xm, ym, ix, iy, tol=0.1):
    """Masque bool : True si (xm, ym) est à moins de `tol` d'un point trivial."""
    if len(ix) == 0: return np.zeros(len(xm), dtype=bool)
    d = np.hypot(xm[:, None] - ix, ym[:, None] - iy)
    return np.any(d < tol, axis=1)


def compute_complex_point(args):
    """Task de pool : ζ(x + i·y) en mpmath (NaN si non convergé)."""
    x, y = args
    try:
        return complex(mp.zeta(mp.mpc(x, y)))
    except Exception:
        return np.nan + 1j * np.nan


def compute_complex_zeta(x, y):
    """Calcule ζ sur la grille entière via un Pool de processeurs."""
    # Générateur paresseux (pas de liste de tuples en mémoire) :
    # même parcours (yj-major, xj-minor) que la comprehension d'origine.
    coords = ((xj, yj) for yj in y for xj in x)
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_complex_point, coords), total=len(y) * len(x), desc="Calcul complex"))
    Z = np.array(results, dtype=np.complex128).reshape(len(y), len(x))
    return Z


def compute_split_point(args):
    """Task de pool : ζ en split-complex via ζ(x+y) et ζ(x−y).

    Rappel : pour un split-complex s = a + b·j, ζ(s) se déduit des deux
    valeurs réelles ζ(a+b) et ζ(a−b), d'où la partie « réelle »
    (ζ(a+b)+ζ(a−b))/2 et la partie « hyperbolique » (ζ(a+b)−ζ(a−b))/2.
    """
    xj, yi = args
    try:
        zp = float(mp.zeta(xj + yi))
        zm = float(mp.zeta(xj - yi))
        s = SplitComplex((zp + zm) / 2, (zp - zm) / 2)
        return s.a, s.b
    except Exception:
        return np.nan, np.nan


def compute_split_zeta(x, y):
    """Calcule les deux composantes (U, V) de ζ en algèbre split."""
    # Générateur paresseux : même ordre (yi-major, xj-minor) qu'à l'origine.
    coords = ((xj, yi) for yi in y for xj in x)
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_split_point, coords), total=len(y) * len(x), desc="Calcul split"))
    zp, zm = zip(*results)
    zp, zm = np.array(zp).reshape(len(y), len(x)), np.array(zm).reshape(len(y), len(x))
    return (zp + zm) / 2, (zp - zm) / 2


def compute_dual_point(xi):
    """Task de pool : (ζ(x), ζ'(x)) en dual — la dérivée vient de mpmath.diff."""
    try:
        d = Dual(float(mp.zeta(xi)), float(mp.diff(mp.zeta, xi)))
        return d.a, d.b
    except Exception:
        return np.nan, np.nan


def compute_dual_zeta(x, y):
    """Calcule ζ(x) (constant selon y) et ζ'(x) (multiplié par l'ordonnée).

    En algèbre dual, ζ(x + y·ε) = ζ(x) + y·ζ'(x)·ε : la dépendance en y est
    linéaire, d'où le np.outer(y, dz).
    """
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_dual_point, x), total=len(x), desc="Calcul dual"))
    z_vals, dz_vals = zip(*results)
    return np.tile(z_vals, (len(y), 1)), np.outer(y, dz_vals)


def save_plot(fig, filename):
    """Sauvegarde une figure unique puis la ferme (évite les fuites mémoire)."""
    fig.tight_layout()
    fig.savefig(filename, dpi=500)
    plt.close(fig)


def generate_all_and_display(selected_modes):
    """Boucle principale : pour chaque mode, calcule, trace et exporte.

    Deux figures par mode : en haut l'image colorée de ζ, en bas les minima
    locaux (bleu = trivial, rouge = non trivial). Des copies PNG individuelles
    sont écrites dans `zeta_images/` et `zeta_minima/`, et les zéros non
    triviaux avec y ≥ 0 sont exportés en CSV.
    """
    fig, axs = plt.subplots(2, len(selected_modes), figsize=(6 * len(selected_modes), 10))
    if len(selected_modes) == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    plt.subplots_adjust(hspace=0.3)
    global_start = time.time()

    for col, key in enumerate(selected_modes):
        print(f"\n=== Traitement de l'ensemble '{key}' ===")
        start = time.time()
        (x0, x1, y0, y1), ppu = config[key]['range'], config[key]['ppu']
        (X, Y), x_vals, y_vals = create_grid(x0, x1, y0, y1, ppu)

        print(" → Calcul des valeurs de la fonction zêta...")
        if key == 'complex':
            Z = compute_complex_zeta(x_vals, y_vals)
            hue = (np.angle(Z) + np.pi) / (2 * np.pi)
            val = compute_log_magnitude(Z)
            data = np.abs(Z)
        elif key == 'split':
            U, V = compute_split_zeta(x_vals, y_vals)
            hue = (np.arctan2(V, U) + np.pi) / (2 * np.pi)
            data = U ** 2 - V ** 2
            val = compute_log_magnitude(data)
        else:
            U, V = compute_dual_zeta(x_vals, y_vals)
            hue = (np.arctan2(V, U) + np.pi) / (2 * np.pi)
            data, val = U, compute_log_magnitude(U)

        print(" → Recherche des minima locaux...")
        minima = find_local_minima(np.abs(data), border=1)
        y_idx, x_idx = np.where(minima)
        x_min, y_min = x_vals[x_idx], y_vals[y_idx]

        print(" → Classification des minima (triviaux / non triviaux)...")
        if key == 'split':
            ix, iy = plot_trivial_lines_split(x_vals[0], x_vals[-1], y_vals[0], y_vals[-1])
            triv_mask = classify_minima_split(x_min, y_min, ix, iy)
        else:
            x_targets = np.arange(np.floor(x0), np.ceil(min(0, x1)) + 1, 2)
            triv_mask = np.isclose(x_min[:, None], x_targets, atol=0.1).any(1)
            if key == 'complex':
                triv_mask &= np.abs(y_min) < 0.1

        x_triv, y_triv = x_min[triv_mask], y_min[triv_mask]
        x_ntriv, y_ntriv = x_min[~triv_mask], y_min[~triv_mask]

        print(" → Génération et sauvegarde des images...")
        img = hsv_to_rgb(hue, val)
        axs[0, col].imshow(img, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], origin='lower', aspect='auto')
        axs[0, col].set(title=f'{key.capitalize()} ζ', xlabel='x', ylabel='y')

        axs[1, col].set_facecolor('white')
        axs[1, col].plot(x_triv, y_triv, 'bo', markersize=1, label='Trivial')
        axs[1, col].plot(x_ntriv, y_ntriv, 'ro', markersize=1, label='Non-trivial')
        axs[1, col].set(title=f'{key.capitalize()} – Minimums locaux', xlabel='x', ylabel='y')
        axs[1, col].set_xlim(x_vals[0], x_vals[-1])
        axs[1, col].set_ylim(y_vals[0], y_vals[-1])
        axs[1, col].legend(fontsize='small')

        for suffix, points in [('func', img), ('minima', (x_triv, y_triv, x_ntriv, y_ntriv))]:
            fig_tmp, ax = plt.subplots(figsize=(6, 5))
            if suffix == 'func':
                ax.imshow(points, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], origin='lower', aspect='auto')
            else:
                ax.set_facecolor('white')
                ax.plot(points[0], points[1], 'bo', markersize=1)
                ax.plot(points[2], points[3], 'ro', markersize=1)
            ax.set_xlim(x_vals[0], x_vals[-1])
            ax.set_ylim(y_vals[0], y_vals[-1])
            ax.set(title=f'{key.capitalize()} – {suffix}', xlabel='x', ylabel='y')
            save_plot(fig_tmp, f"{'zeta_images' if suffix == 'func' else 'zeta_minima'}/{key}_{suffix}.png")

        print(" → Sauvegarde des zéros non triviaux (csv)...")
        filtered = np.column_stack((x_ntriv, y_ntriv))
        filtered = filtered[filtered[:, 1] >= 0]

        if not len(filtered):
            print(f" ⚠ Aucun zéro non trivial y ≥ 0 pour {key}")
        else:
            rounded = np.round(filtered, 8)
            unique_y = np.unique(rounded[:, 1])
            final_points = []
            for y in unique_y:
                same_y = filtered[np.abs(filtered[:, 1] - y) < 1e-8]
                xs = same_y[:, 0]
                x_sel = xs[np.abs(xs) < 1e-8][0] if np.any(np.abs(xs) < 1e-8) else xs[xs > 0][0] if np.any(xs > 0) else xs[0]
                final_points.append([x_sel, y])
            np.savetxt(f'zeta_minima/nontrivial_{key}.csv', final_points, delimiter=',', header='x,y', fmt='%.10f', comments='')
            print(f" ✔ Fichier CSV enregistré : nontrivial_{key}.csv")

        print(f" ✔ Fichiers pour '{key}' générés en {time.time() - start:.2f}s")

    print(f"\n=== Terminé. Temps total : {time.time() - global_start:.2f}s ===")
    plt.show()


def main():
    """Interface interactive : demande les modes à calculer puis lance tout."""
    print("=== Sélection des ensembles à calculer ===")
    print("Entrez une ou plusieurs options parmi : complex, split, dual")
    user_input = input("Votre sélection : ").strip().lower().split()
    valid_modes = [mode for mode in user_input if mode in config]
    if not valid_modes:
        print("Aucun mode valide sélectionné. Exécution annulée.")
        return
    print(f"Modes choisis : {valid_modes}")
    generate_all_and_display(valid_modes)


if __name__ == '__main__':
    main()

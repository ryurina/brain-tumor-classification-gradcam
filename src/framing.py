"""Cadrage des images : part de l'image occupée par le fond noir (aucun PyTorch nécessaire).

Sert à comparer la distribution du cadrage entre l'ENTRAÎNEMENT et le TEST, par classe :
si les images au cadrage serré (peu de fond) sont rares à l'entraînement mais fréquentes au test,
cela peut expliquer les erreurs concentrées sur ces images (voir le rapport, section 3.6).

    python src/framing.py                      # lit dataset/Training et dataset/Testing
    python src/framing.py --threshold 0.393    # seuil « serré » (1er quartile des gliomes du test)

Sorties : results/framing_all.csv et un tableau par classe et par ensemble.
Le masque couvre toute la tête (crâne et visage compris) : il distingue le fond, pas le cerveau.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

EXTS = {".jpg", ".jpeg", ".png"}


def load_gray(path, size=224):
    """Image brute en niveaux de gris, taille size x size, valeurs dans [0,1]."""
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32) / 255.0


def brain_mask(img01, thr=0.05):
    """Masque du premier plan (TÊTE entière) : seuil, ouverture, remplissage des trous.
    À vérifier visuellement ; ajuster thr si le fond n'est pas parfaitement noir."""
    m = img01 > thr
    m = ndimage.binary_opening(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def off_area(path):
    """Part de l'image située hors du masque (fond)."""
    return float((~brain_mask(load_gray(path))).mean())


def scan_split(split_dir, split_name):
    rows = []
    for cls_dir in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in EXTS:
                rows.append({"ensemble": split_name, "classe": cls_dir.name, "image": f.name, "path": str(f)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--threshold", type=float, default=0.393,
                        help="off_area <= seuil : image « serrée » (0,393 = 1er quartile des gliomes du test)")
    args = parser.parse_args()

    from tqdm import tqdm

    data_dir = Path(args.data_dir)
    df = pd.concat([scan_split(data_dir / "Training", "Training"),
                    scan_split(data_dir / "Testing", "Testing")], ignore_index=True)
    print(f"{len(df)} images à mesurer")
    df["off_area"] = [off_area(p) for p in tqdm(df["path"])]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.drop(columns="path").to_csv(results_dir / "framing_all.csv", index=False)

    df["serree"] = df["off_area"] <= args.threshold
    g = df.groupby(["classe", "ensemble"]).agg(
        n=("off_area", "size"), q25=("off_area", lambda s: s.quantile(.25)),
        mediane=("off_area", "median"), q75=("off_area", lambda s: s.quantile(.75)),
        part_serree_pct=("serree", lambda s: 100 * s.mean())).round(3)
    print("\nCadrage (part de fond) par classe et par ensemble :")
    print(g)
    print(f"\n« Serrée » = part de fond <= {args.threshold}. Au test, la part de gliomes serrés vaut ~25 % par construction.")
    print("Comparez les lignes Training et Testing de chaque classe : un grand écart indique un décalage de cadrage.")


if __name__ == "__main__":
    main()

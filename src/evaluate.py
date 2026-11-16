"""Évaluation d'un modèle sur le test : exactitude, rapport par classe, matrice de confusion, logits.

Exemples (depuis la racine du dépôt) :
    python src/evaluate.py --model resnet18
    python src/evaluate.py --model cnn
    python src/evaluate.py --model resnet18 --compare_logits results/logits_resnet18.npz   # vérification
    python src/evaluate.py --model resnet18 --save_logits results/logits_resnet18_bis.npz

Résultats attendus avec les checkpoints du rapport : exactitude test 0,9431 (resnet18) et 0,8675 (cnn).
`--compare_logits` recalcule les logits et les compare à un fichier existant : c'est la preuve que
l'évaluation est reproductible à partir du checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset import get_loaders
from model import build_model, load_checkpoint
from utils import MODELS, get_device


@torch.no_grad()
def get_logits(model, loader, device):
    """Logits et étiquettes d'un DataLoader (même fonction que dans les notebooks)."""
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def plot_confusion(cm, classes, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    except ImportError:  # seaborn absent : repli sur matplotlib
        plt.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
    plt.xlabel("Prediction Label")
    plt.ylabel("Correct Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--checkpoint", default=None, help="défaut : models/<checkpoint du modèle>")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4, help="mettre 0 sous Windows en cas d'erreur")
    parser.add_argument("--save_logits", default=None, help="chemin .npz où sauvegarder les logits val et test")
    parser.add_argument("--compare_logits", default=None, help="fichier .npz existant à comparer aux logits recalculés")
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()

    info = MODELS[args.model]
    device = get_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(args.models_dir) / info["checkpoint"]
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint introuvable : {ckpt} (lancez d'abord src/train.py ou donnez --checkpoint)")

    _, val_loader, test_loader, d = get_loaders(args.data_dir, args.model, args.batch_size, args.num_workers)
    classes = d["classes"]

    model = build_model(args.model, num_classes=len(classes), pretrained=False).to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()
    print(f"Modèle {info['display']} chargé depuis {ckpt} | appareil : {device}")

    test_logits, test_y = get_logits(model, test_loader, device)
    test_pred = test_logits.argmax(1)

    acc = accuracy_score(test_y, test_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%  ({int((test_pred != test_y).sum())} erreurs sur {len(test_y)})")
    print(classification_report(test_y, test_pred, target_names=classes, digits=4, zero_division=0))
    cm = confusion_matrix(test_y, test_pred)
    print("Matrice de confusion (lignes : vraie classe) :")
    print(cm)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(test_y, test_pred, target_names=classes, digits=4, output_dict=True, zero_division=0)
    (results_dir / f"metrics_{args.model}.json").write_text(
        json.dumps({"test_accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}, indent=1)
    )
    if not args.no_figures:
        (results_dir / "figures").mkdir(exist_ok=True)
        plot_confusion(cm, classes, f"{info['display']} - Test Matrice de Confusion",
                       results_dir / "figures" / f"confusion_{args.model}.png")

    if args.save_logits or args.compare_logits:
        val_logits, val_y = get_logits(model, val_loader, device)

    if args.save_logits:
        np.savez(args.save_logits, val_logits=val_logits, val_y=val_y, test_logits=test_logits, test_y=test_y)
        print(f"\nLogits sauvegardés dans {args.save_logits}")

    if args.compare_logits:
        ref = np.load(args.compare_logits)
        same_labels = np.array_equal(ref["test_y"], test_y) and np.array_equal(ref["val_y"], val_y)
        d_test = float(np.abs(ref["test_logits"] - test_logits).max())
        d_val = float(np.abs(ref["val_logits"] - val_logits).max())
        same_pred = np.array_equal(ref["test_logits"].argmax(1), test_pred)
        print(f"\nComparaison avec {args.compare_logits} :")
        print(f"  étiquettes identiques : {same_labels} | prédictions test identiques : {same_pred}")
        print(f"  écart maximal des logits : test {d_test:.2e} | validation {d_val:.2e}")
        ok = same_labels and same_pred and max(d_test, d_val) < 1e-3
        print("  => REPRODUIT" if ok else "  => DIFFÉRENCES (autre checkpoint, autre découpage ou autre version de PyTorch ?)")


if __name__ == "__main__":
    main()

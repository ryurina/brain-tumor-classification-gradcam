"""Entraînement d'un modèle (ResNet18 ou CNN from scratch).

Exemples (depuis la racine du dépôt) :
    python src/train.py --model resnet18                 # 10 époques, lr 1e-4 (comme le notebook)
    python src/train.py --model cnn                      # 30 époques, lr 1e-4 (comme le notebook)
    python src/train.py --model cnn --tag essai2         # écrit models/best_cnn_scratch_essai2.pth

Le checkpoint conservé est celui de meilleure exactitude de validation.
Sécurité : le script REFUSE d'écraser un checkpoint existant (celui des résultats du rapport)
sauf avec --overwrite ; utilisez --tag pour un nouvel essai.
Note : les résultats du rapport ont été obtenus sans fixer les graines PyTorch ; avec --seed 42
l'entraînement est plus reproductible, mais les scores ne seront pas identiques à ceux du rapport.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_loaders
from model import build_model, count_parameters
from utils import MODELS, get_device, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for b, (images, labels) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and b >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total


def validate(model, loader, criterion, device, max_batches=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for b, (images, labels) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--data_dir", default="dataset", help="dossier contenant Training/ et Testing/")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--epochs", type=int, default=None, help="défaut : 10 (resnet18) ou 30 (cnn)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4, help="mettre 0 sous Windows en cas d'erreur")
    parser.add_argument("--seed", type=int, default=42, help="graine PyTorch/NumPy (le découpage train/val utilise toujours 42)")
    parser.add_argument("--tag", default="", help="suffixe ajouté aux fichiers de sortie")
    parser.add_argument("--overwrite", action="store_true", help="autorise l'écrasement du checkpoint")
    parser.add_argument("--no_pretrained", action="store_true", help="ResNet18 sans poids ImageNet (tests)")
    parser.add_argument("--limit_batches", type=int, default=None, help="n'utiliser que N lots par époque (test rapide)")
    args = parser.parse_args()

    info = MODELS[args.model]
    epochs = args.epochs if args.epochs is not None else info["epochs"]
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_path = Path(args.models_dir) / info["checkpoint"].replace(".pth", f"{suffix}.pth")
    hist_path = Path(args.results_dir) / f"history_{args.model}{suffix}.json"

    if ckpt_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Le checkpoint {ckpt_path} existe déjà : il n'est pas écrasé.\n"
            f"Utilisez --tag NOM pour un nouvel essai, ou --overwrite pour le remplacer."
        )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, _, d = get_loaders(args.data_dir, args.model, args.batch_size, args.num_workers)
    print("Training:", len(d["train"]), "| Validation:", len(d["val"]), "| Test:", len(d["test"]))
    print("Classes:", d["classes"])

    model = build_model(args.model, num_classes=len(d["classes"]), pretrained=not args.no_pretrained).to(device)
    print(f"Modèle : {args.model} | paramètres entraînables : {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.limit_batches)
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.limit_batches)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print("Enregistrement du meilleur model")

    hist_path.write_text(json.dumps(history, indent=1))
    print("Entraînement complete!")
    print(f"Validation Accuracy: {best_val_acc:.2f}% | durée : {(time.time() - t0) / 60:.1f} min")
    print(f"Checkpoint : {ckpt_path} | historique : {hist_path}")


if __name__ == "__main__":
    main()

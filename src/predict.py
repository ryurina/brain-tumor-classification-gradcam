"""Prédiction sur une image : classe, confiance, et carte Grad-CAM (optionnelle).

    python src/predict.py --model resnet18 --image chemin/vers/irm.jpg
    python src/predict.py --model resnet18 --image irm.jpg --cam sortie.png

AVERTISSEMENT : projet académique, ce n'est PAS un outil de diagnostic médical. Le modèle se trompe,
parfois avec une confiance supérieure à 0,99. Ne jamais l'utiliser pour décider d'une prise en charge.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_transforms
from gradcam import GradCAM, denormalize
from model import build_model, get_target_layer, load_checkpoint
from utils import CLASSES, MODELS, get_device

DISCLAIMER = ("AVERTISSEMENT : étude académique, pas un outil de diagnostic médical. "
              "Ne pas utiliser pour une décision clinique.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cam", default=None, help="chemin de sortie d'une figure Grad-CAM (.png)")
    args = parser.parse_args()

    info = MODELS[args.model]
    device = get_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(args.models_dir) / info["checkpoint"]
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint introuvable : {ckpt}")

    model = build_model(args.model, num_classes=len(CLASSES), pretrained=False).to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()

    _, eval_tf = get_transforms(args.model)
    img = Image.open(args.image).convert("RGB")      # ImageFolder charge aussi en RGB
    x = eval_tf(img).unsqueeze(0).to(device)

    print(DISCLAIMER)
    cam_obj = GradCAM(model, get_target_layer(model, args.model))
    cam, out = cam_obj.generate(x)
    probs = F.softmax(out.detach(), dim=1)[0].cpu()
    pred = int(probs.argmax())
    print(f"\nPrédiction : {CLASSES[pred]} (confiance {probs[pred]:.3f})")
    for k, c in enumerate(CLASSES):
        print(f"  {c:11s} {probs[k]:.3f}")

    if args.cam:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        base = denormalize(x[0], args.model)
        cmap = None if base.ndim == 3 else "gray"
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        axes[0].imshow(base, cmap=cmap); axes[0].set_title("Image")
        axes[1].imshow(base, cmap=cmap); axes[1].imshow(cam, cmap="jet", alpha=0.4)
        axes[1].set_title(f"Grad-CAM : {CLASSES[pred]} ({probs[pred]:.2f})")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(DISCLAIMER, fontsize=8, color="red")
        plt.tight_layout()
        plt.savefig(args.cam, dpi=150)
        plt.close()
        print(f"Figure sauvegardée : {args.cam}")


if __name__ == "__main__":
    main()

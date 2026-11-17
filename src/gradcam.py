import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from dataset import IMAGENET_MEAN, IMAGENET_STD, get_loaders
from model import build_model, get_target_layer, load_checkpoint
from utils import MODELS, get_device


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None

        # Récupère les activations de la couche cible
        self.handle = target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, class_idx=None, signed=False):
        """Retourne (carte 2D à la taille de l'image, sorties du modèle).

        class_idx=None : classe prédite. Donner un entier pour expliquer une autre classe (ex. la vraie).
        signed=False : carte ReLU normalisée dans [0,1] (Grad-CAM classique).
        signed=True  : carte SANS ReLU, divisée par son maximum en valeur absolue (dans [-1,1]) :
                       positif = soutient la classe, négatif = va contre elle.
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]

        # Gradient du score par rapport aux activations
        gradients = torch.autograd.grad(outputs=score, inputs=self.activations)[0]

        # Moyenne spatiale des gradients par canal, puis combinaison pondérée des cartes
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        if not signed:
            cam = F.relu(cam)

        # Redimensionnement à la taille de l'image (carte 7x7 -> 224x224, interpolation bilinéaire)
        cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        # Normalisation (même ordre que le notebook 07, dont viennent les statistiques du rapport)
        if signed:
            cam = cam / (cam.abs().max() + 1e-8)       # dans [-1, 1]
        else:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)             # dans [0, 1]

        return cam.detach().cpu().numpy(), output

    def remove_hook(self):
        self.handle.remove()


def denormalize(x, model_name):
    """Tenseur normalisé (C,H,W) -> image affichable (H,W) pour le CNN, (H,W,3) pour ResNet18."""
    x = x.detach().cpu()
    if model_name == "resnet18":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return np.clip((x * std + mean).permute(1, 2, 0).numpy(), 0, 1)
    return np.clip((x * 0.5 + 0.5).squeeze(0).numpy(), 0, 1)


def load_gray(path, size=224):
    """Image brute en niveaux de gris, taille size x size, valeurs dans [0,1]."""
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32) / 255.0


def brain_mask(img01, thr=0.05):
    """Masque du premier plan : TÊTE entière (crâne et visage compris), pas seulement le cerveau.
    Seuil, ouverture morphologique, remplissage des trous. À vérifier visuellement ; ajuster thr si le fond
    n'est pas parfaitement noir."""
    m = img01 > thr
    m = ndimage.binary_opening(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def error_groups(results_dir):
    """Groupes d'images de test construits à partir des logits sauvegardés (results/logits_*.npz).

    Retourne (y, pred, conf, groupes) où groupes est un dictionnaire nom -> indices d'images de test.
    """
    d = {m: np.load(Path(results_dir) / MODELS[m]["logits"]) for m in MODELS}
    y = d["resnet18"]["test_y"]
    p = {m: _softmax(d[m]["test_logits"]) for m in d}
    pred = {m: p[m].argmax(1) for m in d}
    conf = {m: p[m].max(1) for m in d}
    wrong = {m: pred[m] != y for m in d}
    groups = {
        "erreurs_communes": np.where(wrong["resnet18"] & wrong["cnn"])[0],
        "resnet18_erreurs_confiantes": np.where(wrong["resnet18"] & (conf["resnet18"] >= 0.9))[0],
        "resnet18_glioma_vers_notumor": np.where((y == 0) & (pred["resnet18"] == 2))[0],
        "cnn_glioma_vers_notumor": np.where((y == 0) & (pred["cnn"] == 2))[0],
        "temoins_bien_classees": np.where(~wrong["resnet18"] & ~wrong["cnn"])[0],
    }
    return y, pred, conf, groups


def compute_cam_stats(cam_obj, dataset, indices, device):
    """Pour chaque image : classe vraie/prédite, confiance, off_cam (part de la carte sur le fond), off_area."""
    from tqdm import tqdm

    rows = []
    for i in tqdm(indices):
        x, y = dataset[i]
        cam, out = cam_obj.generate(x.unsqueeze(0).to(device))
        p = F.softmax(out, dim=1)[0].detach().cpu().numpy()
        img = load_gray(dataset.samples[i][0])
        m = brain_mask(img)
        rows.append({
            "idx": int(i), "y": int(y), "pred": int(p.argmax()), "conf": float(p.max()),
            "off_cam": float(cam[~m].sum() / (cam.sum() + 1e-8)),
            "off_area": float((~m).mean()),
        })
    return pd.DataFrame(rows)


def summarize(results_dir):
    """Médianes de off_cam et off_area par modèle, classe vraie et bonne/mauvaise prédiction.
    Lit results/gradcam_stats_<modèle>.csv et écrit results/gradcam_summary.csv."""
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    rows = []
    for m in MODELS.values():
        path = Path(results_dir) / f"gradcam_stats_{m['display']}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["bien classée"] = df["pred"] == df["y"]
        df["classe vraie"] = [classes[k] for k in df["y"]]
        for (cls, ok), g in df.groupby(["classe vraie", "bien classée"]):
            rows.append({"modèle": m["display"], "classe vraie": cls, "bien classée": ok, "n": len(g),
                         "part CAM hors tête (médiane)": g["off_cam"].median(),
                         "part image hors tête (médiane)": g["off_area"].median()})
    if not rows:
        raise SystemExit(f"Aucun fichier gradcam_stats_*.csv dans {results_dir} (lancez d'abord : gradcam.py stats)")
    out = pd.DataFrame(rows).round(3)
    out.to_csv(Path(results_dir) / "gradcam_summary.csv", index=False)
    print(out.to_string(index=False))
    report_tables(results_dir)


def report_tables(results_dir):
    """Les deux tableaux du rapport, recalculés depuis results/gradcam_stats_*.csv :
    (1) bien classées / erreurs par modèle ; (2) par classe vraie."""
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    stats = {m["display"]: pd.read_csv(Path(results_dir) / f"gradcam_stats_{m['display']}.csv") for m in MODELS.values()
             if (Path(results_dir) / f"gradcam_stats_{m['display']}.csv").exists()}
    pooled = []
    for name, d in stats.items():
        d = d.assign(diff=d["off_cam"] - d["off_area"], ok=d["pred"] == d["y"])
        for lab, g in [("bien classées", d[d["ok"]]), ("erreurs", d[~d["ok"]])]:
            pooled.append({"modèle": name, "groupe": lab, "n": len(g), "off_cam (méd.)": g["off_cam"].median(),
                           "off_area (méd.)": g["off_area"].median(), "diff. médiane": g["diff"].median(),
                           "% images off_cam > off_area": 100 * (g["diff"] > 0).mean()})
    pooled = pd.DataFrame(pooled).round(3)
    pooled.to_csv(Path(results_dir) / "gradcam_pooled.csv", index=False)
    print("\nPart de la carte sur le fond, bien classées contre erreurs :")
    print(pooled.to_string(index=False))

    ref = next(iter(stats.values()))
    by_class = pd.DataFrame({"classe": classes,
                             "off_area (méd.)": [ref.loc[ref["y"] == k, "off_area"].median() for k in range(4)]})
    for name, d in stats.items():
        by_class[f"off_cam {name} (méd.)"] = [d.loc[d["y"] == k, "off_cam"].median() for k in range(4)]
        by_class[f"{name} : % off_cam > off_area"] = [100 * (d.loc[d["y"] == k, "off_cam"] > d.loc[d["y"] == k, "off_area"]).mean()
                                                    for k in range(4)]
    by_class = by_class.round(3)
    by_class.to_csv(Path(results_dir) / "gradcam_by_class.csv", index=False)
    print("\nPar classe vraie :")
    print(by_class.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["overlay", "stats", "summary", "groups"])
    parser.add_argument("--model", choices=list(MODELS), default=None, help="obligatoire pour overlay et stats")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="indices d'images de test (overlay)")
    parser.add_argument("--group", default=None, help="tirer les images dans un groupe (voir le mode groups)")
    parser.add_argument("--n", type=int, default=6, help="nombre d'images tirées au hasard si --indices absent")
    parser.add_argument("--class_idx", type=int, default=None, help="classe à expliquer (défaut : la classe prédite)")
    parser.add_argument("--true_class", action="store_true", help="expliquer la VRAIE classe de chaque image")
    parser.add_argument("--signed", action="store_true", help="carte sans ReLU : positif = pour la classe, négatif = contre")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="stats : se limiter aux N premières images")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "summary":
        summarize(results_dir)
        return

    if args.mode == "groups":
        y, pred, conf, groups = error_groups(results_dir)
        rows = []
        for name, idx in groups.items():
            print(f"{name:30s} {len(idx):5d} images")
            rows += [{"groupe": name, "idx": int(i)} for i in idx]
        pd.DataFrame(rows).to_csv(results_dir / "gradcam_groups.csv", index=False)
        print(f"\nIndices sauvegardés dans {results_dir / 'gradcam_groups.csv'}")
        return

    if args.model is None:
        raise SystemExit("--model est obligatoire pour les modes overlay et stats")

    info = MODELS[args.model]
    device = get_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(args.models_dir) / info["checkpoint"]
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint introuvable : {ckpt}")

    _, _, _, d = get_loaders(args.data_dir, args.model, 32, args.num_workers)
    test_ds, classes = d["test"], d["classes"]

    model = build_model(args.model, num_classes=len(classes), pretrained=False).to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()
    cam_obj = GradCAM(model, get_target_layer(model, args.model))

    if args.mode == "stats":
        idx = list(range(len(test_ds)))[: args.limit] if args.limit else list(range(len(test_ds)))
        df = compute_cam_stats(cam_obj, test_ds, idx, device)
        out = results_dir / f"gradcam_stats_{info['display']}.csv"
        df.assign(modele=info["display"]).to_csv(out, index=False)
        df["bien_classee"] = df["pred"] == df["y"]
        df["classe"] = [classes[k] for k in df["y"]]
        print(f"\nStatistiques sauvegardées dans {out}")
        print(df.groupby(["classe", "bien_classee"])[["off_cam", "off_area"]].median().round(3))
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.indices:
        indices = args.indices
    elif args.group:
        _, _, _, groups = error_groups(results_dir)
        if args.group not in groups:
            raise SystemExit(f"Groupe inconnu : {args.group}. Choix : {list(groups)}")
        pool = groups[args.group]
        indices = np.random.default_rng(args.seed).choice(pool, min(args.n, len(pool)), replace=False).tolist()
    else:
        indices = np.random.default_rng(args.seed).choice(len(test_ds), args.n, replace=False).tolist()

    fig, axes = plt.subplots(len(indices), 3, figsize=(10, 3.3 * len(indices)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(indices):
        x, y = test_ds[i]
        ci = int(y) if args.true_class else args.class_idx
        cam, out = cam_obj.generate(x.unsqueeze(0).to(device), class_idx=ci, signed=args.signed)
        pred = out.argmax(dim=1).item()
        conf = F.softmax(out, dim=1)[0, pred].item()
        img = denormalize(x, args.model)
        cmap = None if img.ndim == 3 else "gray"
        cam_kw = dict(cmap="seismic", vmin=-1, vmax=1) if args.signed else dict(cmap="jet")

        axes[r, 0].imshow(img, cmap=cmap)
        axes[r, 0].set_title(f"image {i}", fontsize=9)
        axes[r, 1].imshow(cam, **cam_kw)
        axes[r, 1].set_title("Carte signée" if args.signed else "Carte Grad-CAM", fontsize=9)
        axes[r, 2].imshow(img, cmap=cmap)
        axes[r, 2].imshow(cam, alpha=0.45 if args.signed else 0.4, **cam_kw)
        shown = classes[ci] if ci is not None else classes[pred]
        axes[r, 2].set_title(f"vrai : {classes[y]} | prédit : {classes[pred]} ({conf:.2f})\ncarte de : {shown}",
                             fontsize=9, color="green" if pred == y else "red")
        for ax in axes[r]:
            ax.axis("off")
    plt.suptitle(f"{info['display']} - Grad-CAM" + (" signé (rouge : pour la classe, bleu : contre)" if args.signed else ""), y=1.0)
    plt.tight_layout()
    (results_dir / "figures").mkdir(exist_ok=True)
    tag = ("_vraie" if args.true_class else "") + ("_signee" if args.signed else "")
    out = results_dir / "figures" / f"gradcam_{args.model}{tag}_overlay.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Figure sauvegardée : {out} (images {indices})")


if __name__ == "__main__":
    main()

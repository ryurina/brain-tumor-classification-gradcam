import random

import numpy as np

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # ordre alphabétique d'ImageFolder

# Noms de fichiers identiques à ceux des notebooks (dossiers models/ et results/)
MODELS = {
    "resnet18": {
        "display": "ResNet18",
        "checkpoint": "best_resnet18.pth",
        "logits": "logits_resnet18.npz",
        "epochs": 10,     # valeur utilisée dans model_resnet18.ipynb
    },
    "cnn": {
        "display": "CNN",
        "checkpoint": "best_cnn_scratch.pth",
        "logits": "logits_cnn.npz",
        "epochs": 30,     # valeur utilisée dans model_cnn_scratch.ipynb
    },
}


def set_seed(seed: int):
    """Fixe les graines Python, NumPy et PyTorch.

    Les résultats du rapport ont été obtenus SANS fixer les graines PyTorch :
    relancer l'entraînement donne donc des scores proches, pas identiques.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

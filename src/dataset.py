from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(model_name: str):
    """Retourne (transformation d'entraînement, transformation d'évaluation).

    resnet18 : RGB 224x224, normalisation ImageNet.
    cnn      : niveaux de gris (1 canal) 224x224, moyenne 0,5 / écart-type 0,5.
    """
    if model_name == "resnet18":
        mean, std, pre = IMAGENET_MEAN, IMAGENET_STD, []
    elif model_name == "cnn":
        mean, std = [0.5], [0.5]
        pre = [transforms.Grayscale(num_output_channels=1)]
    else:
        raise ValueError(f"Modèle inconnu : {model_name} (choisir 'resnet18' ou 'cnn')")

    train_tf = transforms.Compose(
        pre + [transforms.Resize((224, 224)), transforms.RandomRotation(10),
               transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    eval_tf = transforms.Compose(
        pre + [transforms.Resize((224, 224)),
               transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return train_tf, eval_tf


def build_datasets(data_dir, model_name: str, val_size: float = 0.2, split_seed: int = 42):
    """Construit les ensembles d'entraînement, de validation et de test.

    Deux ImageFolder sur le même dossier Training (avec et sans augmentation) permettent
    d'avoir une validation sans augmentation, avec les mêmes indices de découpage.
    """
    data_dir = Path(data_dir)
    train_tf, eval_tf = get_transforms(model_name)

    train_aug = datasets.ImageFolder(data_dir / "Training", transform=train_tf)
    full_train_eval = datasets.ImageFolder(data_dir / "Training", transform=eval_tf)
    test_dataset = datasets.ImageFolder(data_dir / "Testing", transform=eval_tf)

    targets = train_aug.targets
    indices = list(range(len(targets)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, stratify=targets, random_state=split_seed
    )
    return {
        "train": Subset(train_aug, train_idx),
        "val": Subset(full_train_eval, val_idx),
        "test": test_dataset,
        "classes": train_aug.classes,
        "train_targets": train_aug.targets,
    }


def get_loaders(data_dir, model_name: str, batch_size: int = 32, num_workers: int = 4,
                split_seed: int = 42):
    """Retourne les DataLoaders (train, val, test) et le dictionnaire des ensembles.

    num_workers=4 comme dans les notebooks ; mettre 0 sous Windows en cas d'erreur de multiprocessing.
    """
    d = build_datasets(data_dir, model_name, split_seed=split_seed)
    train_loader = DataLoader(d["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(d["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(d["test"], batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, d

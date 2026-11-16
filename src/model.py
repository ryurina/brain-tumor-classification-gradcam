"""Les deux modèles comparés : un CNN entraîné de zéro et ResNet18 pré-entraîné."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """CNN from scratch : 5 blocs Conv-BN-ReLU-MaxPool (1 -> 16 -> 32 -> 64 -> 128 -> 256 canaux),
    pooling moyen global, dropout 0,3, couche linéaire. 393 844 paramètres pour 4 classes."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16), conv_block(16, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)


def build_resnet18(num_classes=4, pretrained=True):
    """ResNet18 (poids ImageNet si pretrained=True), dernière couche remplacée, fine-tuning complet."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int = 4, pretrained: bool = True):
    """name : 'resnet18' ou 'cnn'. pretrained ne concerne que ResNet18.

    Pour charger un checkpoint (évaluation, Grad-CAM, prédiction), utiliser pretrained=False :
    inutile de télécharger les poids ImageNet.
    """
    if name == "resnet18":
        return build_resnet18(num_classes, pretrained)
    if name == "cnn":
        return SmallCNN(num_classes)
    raise ValueError(f"Modèle inconnu : {name} (choisir 'resnet18' ou 'cnn')")


def get_target_layer(model, name: str):
    """Couche utilisée par Grad-CAM : cartes de 7x7 dans les deux cas."""
    return model.layer4[-1] if name == "resnet18" else model.features[-1]


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

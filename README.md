# Peut-on faire confiance à un CNN pour classer des IRM cérébrales ?

Comparaison d'un CNN entraîné de zéro et d'un ResNet18 pré-entraîné sur le jeu *Brain Tumor MRI*, avec analyse par **Grad-CAM** et **calibration** des probabilités.

## 1. Objectifs

Un bon score suffit-il pour faire confiance à un modèle ? Sur la classification d'IRM cérébrales en quatre classes (gliome, méningiome, pas de tumeur, tumeur de l'hypophyse), le projet étudie :

1. **Les performances et les erreurs** de deux modèles : un petit CNN from scratch et ResNet18 affiné.
2. **La calibration** : les probabilités annoncées sont-elles honnêtes (ECE, NLL, Brier) ? Le *temperature scaling* les corrige-t-il ?
3. **Grad-CAM** : où regardent les modèles quand ils se trompent ? Existe-t-il un raccourci lié au cadrage de l'image ?

## 2. Résultats principaux

| | ResNet18 | CNN from scratch |
|---|---|---|
| Paramètres | ~11 millions | 393 844 |
| Entraînement | 10 époques, Adam, lr 1e-4 | 30 époques, Adam, lr 1e-4 |
| Meilleure exactitude de validation | 98,57 % | 92,23 % |
| **Exactitude sur le test (1 600 images)** | **94,31 %** | **86,75 %** |
| Rappel du gliome | 0,82 | 0,695 |
| Tumeurs prédites « pas de tumeur » | 30 | 77 |
| ECE test (avant / après *temperature scaling*) | 0,0330 / 0,0330 | 0,0295 / 0,0396 |
| `T*` appris sur la validation | 0,999 | 0,911 |

Enseignements (détails et nuances dans le rapport) :

- **La validation surestime le test** pour les deux modèles (−4,3 et −5,5 points).
- **Le *temperature scaling* appris sur la validation ne corrige pas le test** : le CNN est sous-confiant sur la validation et sur-confiant sur le test, donc la correction va dans le mauvais sens.
- **77 des 91 erreurs de ResNet18 sont aussi des erreurs du CNN.**
- **Les erreurs se concentrent sur les images au cadrage serré** : 51 des 72 gliomes manqués par ResNet18 sont dans le quart de gliomes dont le fond est le plus petit (51 % d'erreurs contre 6 à 9 % ailleurs). ResNet18 regarde la tête plutôt que le fond ; le CNN s'appuie davantage sur le fond, surtout pour les gliomes.
- Ces observations sont **corrélationnelles** : elles ne prouvent pas que les modèles utilisent le cadrage.

## 3. Données

- **Brain Tumor MRI Dataset** (Masoud Nickparvar, Kaggle) : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Classes : `glioma`, `meningioma`, `notumor`, `pituitary` (dossiers par classe).
- Copie utilisée : `Training` = 5 600 images, `Testing` = 1 600 images (400 par classe). La version exacte peut différer de celle décrite sur la page Kaggle.
- `Training` est séparé en entraînement (4 480 images) et validation (1 120 images) par tirage **stratifié** (`random_state=42`). `Testing` n'est utilisé qu'en évaluation finale.
- Pas d'identifiant de patient : la séparation se fait par image, une fuite entre entraînement et validation est possible (non vérifiée).
- **Les données ne sont pas dans ce dépôt.** Téléchargez-les depuis Kaggle et placez-les dans `dataset/Training` et `dataset/Testing`.

## 4. Méthode (résumé)

| | ResNet18 | CNN from scratch |
|---|---|---|
| Entrée | RGB, 224×224, normalisation ImageNet | niveaux de gris (1 canal), 224×224, moyenne 0,5 / écart-type 0,5 |
| Initialisation | poids ImageNet, fine-tuning complet | aléatoire |
| Architecture | ResNet18, dernière couche à 4 sorties | 5 blocs Conv-BatchNorm-ReLU-MaxPool (16→256 canaux), pooling global, dropout 0,3 |
| Augmentation (train seulement) | rotation ±10° | rotation ±10° |

- Perte : entropie croisée, lots de 32, sur CPU. Le checkpoint conservé est celui de meilleure exactitude de validation.
- **Calibration** : ECE (10 intervalles), NLL, score de Brier ; *temperature scaling* appris sur la validation ; diagnostic « oracle » sur le test (non utilisé comme méthode) ; signalement des prédictions de faible confiance.
- **Grad-CAM** sur `layer4[-1]` (ResNet18) et `features[-1]` (CNN), classe prédite, cartes 7×7 agrandies à 224×224. La part de la carte située sur le **fond noir** est comparée à la part de l'image occupée par le fond.

## 5. Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── model_resnet18.ipynb            # entraînement, évaluation, logits de ResNet18
│   ├── model_cnn_scratch.ipynb         # entraînement, évaluation, logits du CNN
├── results/                            # logits (.npz), tableaux (.csv), figures
├── models/                             # checkpoints (.pth), NON versionnés
├── dataset/                            # données Kaggle, NON versionnées
```

## 6. Installation

```bash
git clone https://github.com/ryurina/radiography_pulmonaires_cnn.git
cd radiography_pulmonaires_cnn
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Un GPU n'est pas nécessaire : les résultats ont été obtenus sur CPU. Versions utilisées : Python **À COMPLÉTER**, PyTorch **À COMPLÉTER**, torchvision **À COMPLÉTER**.

## 7. Reproduire les résultats

Placer les données dans `dataset/Training` et `dataset/Testing`, puis lancer les notebooks **dans cet ordre** depuis le dossier `notebooks/` :

| Étape | Notebook | Produit |
|---|---|---|
| 1 | `model_resnet18.ipynb` | `models/best_resnet18.pth`, `results/logits_resnet18.npz` |
| 2 | `model_cnn_scratch.ipynb` | `models/best_cnn_scratch.pth`, `results/logits_cnn.npz` |

Remarques sur la reproductibilité :

- La **séparation entraînement/validation** est reproductible (`random_state=42`).
- Les **graines de PyTorch ne sont pas fixées** : relancer l'entraînement donne des résultats légèrement différents (la validation du CNN varie de ±3 points d'une époque à l'autre). Les chiffres du rapport viennent d'une seule exécution par modèle.
- Les logits de validation et de test sont fournis dans `results/` : l'étape 3 se refait **sans réentraîner**. L'étape 4 demande les checkpoints.

## 8. Limites

- Une seule exécution par modèle ; budgets d'entraînement inégaux (10 contre 30 époques, poids pré-entraînés contre aléatoires). Le CNN n'avait pas convergé : son score est une borne basse.
- Pas d'identifiant de patient ; étiquettes du jeu non auditées ; un seul jeu de données, pas de validation externe.
- Grad-CAM : cartes 7×7 floues ; le masque de mesure couvre toute la tête (crâne et visage compris), il distingue le fond, pas le cerveau ; une carte n'établit pas de causalité.
- La mesure du cadrage mélange zoom, plan de coupe et effet du seuil d'intensité.
- Ce travail n'est pas un outil clinique.

## 9. Références

- M. Nickparvar, *Brain Tumor MRI Dataset*, Kaggle (lien en section 3).
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV 2017.
- C. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
- S. Ioffe, C. Szegedy, *Batch Normalization*, ICML 2015 ; D. Kingma, J. Ba, *Adam*, ICLR 2015 ; A. Paszke et al., *PyTorch*, NeurIPS 2019.

## 10. Auteur

RANARISON Rina — Mathématiques appliquées (Calcul numérique), Université d'Antananarivo.

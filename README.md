# Où regarde un CNN qui classe des IRM cérébrales ? Interprétation par Grad-CAM

Projet final du cours *Réseaux de neurones : vision par ordinateur* (Master 2 Mathématiques appliquées, Université d'Antananarivo).
Comparaison d'un CNN entraîné de zéro et d'un ResNet18 pré-entraîné sur le jeu *Brain Tumor MRI*, avec **Grad-CAM** pour interpréter leurs décisions, en particulier leurs erreurs.

> **Avertissement** : étude académique. Ce n'est pas un outil de diagnostic médical et il ne doit pas être utilisé pour décider d'une prise en charge.

Rapport complet (IMRaD, 12 pages) : [`report/rapport_final.pdf`](report/rapport_final.pdf) — sources LaTeX dans [`report/`](report/).

## 1. Objectifs

1. Comparer les performances et les **erreurs** de deux modèles sur la classification d'IRM cérébrales en quatre classes (gliome, méningiome, pas de tumeur, tumeur de l'hypophyse).
2. **Expliquer Grad-CAM** : ses formules, leur justification, ce qu'une carte signifie et ne signifie pas.
3. **Interpréter** les cartes des deux modèles, sur des exemples et par une mesure sur toutes les images de test.
4. Examiner si un **raccourci lié au cadrage** de l'image peut expliquer certaines erreurs.

## 2. Résultats principaux

| | ResNet18 | CNN from scratch |
|---|---|---|
| Paramètres | ~11 millions | 393 844 |
| Entraînement | 10 époques, Adam, lr 1e-4 | 30 époques, Adam, lr 1e-4 |
| Meilleure exactitude de validation | 98,57 % | 92,23 % |
| **Exactitude sur le test (1 600 images)** | **94,31 %** | **86,75 %** |
| Rappel du gliome | 0,82 | 0,695 |
| Tumeurs prédites « pas de tumeur » | 30 | 77 |
| Part de la carte Grad-CAM sur le fond / part de l'image qui est du fond (médianes) | 0,19 / 0,42 | 0,32 / 0,42 |
| Idem pour les gliomes | 0,24 / 0,49 | **0,49 / 0,49** |

Enseignements (détails et nuances dans le rapport) :

- **77 des 91 erreurs de ResNet18 sont aussi des erreurs du CNN.**
- **ResNet18 regarde la tête, pas le fond** ; le **CNN s'appuie davantage sur le fond**, surtout pour les gliomes.
- **Le cadrage des images (zoom, plan de coupe) dépend de la classe** et suffit seul à retrouver la classe avec 41 à 46 % d'exactitude (hasard : 25 %).
- **Les erreurs se concentrent sur les images au cadrage serré** : 51 des 72 gliomes manqués par ResNet18 sont dans le quart de gliomes dont le fond est le plus petit (51 % d'erreurs contre 6 à 9 % ailleurs).
- Ces observations sont **corrélationnelles** : elles ne prouvent pas que les modèles utilisent le cadrage.

## 3. Grad-CAM en bref

Pour la dernière couche convolutive (cartes d'activation $A^k$ de taille $7\times7$) et une classe $c$ de score $y^c$ (logit) :

```
alpha_k^c = (1/Z) * somme_{i,j} d y^c / d A^k_{ij}        avec Z = 7 x 7 = 49   (importance du canal k)
L^c       = ReLU( somme_k alpha_k^c * A^k )                                    (carte 7 x 7)
```

La carte est ensuite agrandie à 224×224 (interpolation bilinéaire) et normalisée dans [0, 1].

- **Pour ces deux architectures** (pooling moyen global puis couche linéaire de poids `W`), le gradient vaut `W_ck / Z` : Grad-CAM est identique à **CAM** (Zhou et al., 2016) au facteur `1/Z` près. La carte est la contribution locale `s_ij = somme_k W_ck A^k_ij` de chaque position, et le score est la **moyenne** de ces 49 contributions.
- La **ReLU** ne garde que ce qui *soutient* la classe : une zone « froide » peut être sans effet **ou** pousser contre la classe. La carte **signée** (sans ReLU) lève cette ambiguïté (rouge : pour, bleu : contre).
- La carte est **floue** (cases de 32×32 pixels), **normalisée image par image** et **ne prouve pas** la causalité.

Le rapport détaille ces points (section 3) et leur lecture (tableau « ce que la carte indique / n'indique pas »).

## 4. Données

- **Brain Tumor MRI Dataset** (Masoud Nickparvar, Kaggle) : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Classes : `glioma`, `meningioma`, `notumor`, `pituitary` (dossiers par classe).
- Copie utilisée : `Training` = 5 600 images, `Testing` = 1 600 images (400 par classe). La version exacte peut différer de celle décrite sur la page Kaggle.
- `Training` est séparé en entraînement (4 480 images) et validation (1 120 images) par tirage **stratifié** (`random_state=42`). `Testing` n'est utilisé qu'en évaluation finale.
- Pas d'identifiant de patient : la séparation se fait par image, une fuite entre entraînement et validation est possible (non vérifiée).
- **Les données ne sont pas dans ce dépôt.** Téléchargez-les depuis Kaggle et placez-les dans `dataset/Training` et `dataset/Testing`.

## 5. Méthode (résumé)

| | ResNet18 | CNN from scratch |
|---|---|---|
| Entrée | RGB, 224×224, normalisation ImageNet | niveaux de gris (1 canal), 224×224, moyenne 0,5 / écart-type 0,5 |
| Initialisation | poids ImageNet, fine-tuning complet | aléatoire |
| Architecture | ResNet18, dernière couche à 4 sorties | 5 blocs Conv-BatchNorm-ReLU-MaxPool (16→256 canaux), pooling global, dropout 0,3 |
| Augmentation (train seulement) | rotation ±10° | rotation ±10° |

- Perte : entropie croisée, lots de 32, sur CPU. Le checkpoint conservé est celui de meilleure exactitude de validation.
- **Grad-CAM** sur `layer4[-1]` (ResNet18) et `features[-1]` (CNN), classe prédite (ou vraie, avec `--true_class`).
- **Mesure sur le fond** : un masque du premier plan (seuillage, ouverture, remplissage ; il couvre toute la **tête**, pas seulement le cerveau) donne `off_cam` (part de la carte sur le fond) et `off_area` (part de l'image occupée par le fond, proxy du cadrage).

## 6. Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── notebooks/                          # la démarche, avec sorties et figures (à lire pour comprendre)
│   ├── model_resnet18.ipynb            # entraînement, évaluation, logits de ResNet18
│   ├── model_cnn_scratch.ipynb         # entraînement, évaluation, logits du CNN
│   └── 07_gradcam_errors.ipynb         # Grad-CAM sur les erreurs, cartes signées, mesure du fond
├── src/                                # les mêmes étapes en scripts (à lancer pour reproduire / utiliser)
│   ├── utils.py        # noms des modèles et des fichiers, graines, appareil
│   ├── dataset.py      # transformations, découpage 80/20 (random_state=42), DataLoaders
│   ├── model.py        # SmallCNN, ResNet18 à 4 sorties, chargement des checkpoints
│   ├── train.py        # entraînement
│   ├── evaluate.py     # test : métriques, matrice de confusion, logits, vérification
│   ├── gradcam.py      # Grad-CAM, groupes d'erreurs, figures, cartes signées, mesure du fond
│   ├── framing.py      # cadrage des images, entraînement contre test (sans PyTorch)
│   └── predict.py      # prédiction sur une image (classe, confiance, Grad-CAM)
├── results/                            # logits (.npz), tableaux (.csv), figures
├── models/                             # checkpoints (.pth), NON versionnés
├── dataset/                            # données Kaggle, NON versionnées
└── report/                             # rapport LaTeX (main.tex), figures, rapport_final.pdf
```

Les **notebooks** sont la trace des résultats du rapport. Les **scripts** de `src/` reprennent le même code (mêmes transformations, mêmes architectures, mêmes hyperparamètres, même Grad-CAM) pour reproduire et réutiliser, sans notebook.

## 7. Installation

```bash
git clone https://github.com/ryurina/radiography_pulmonaires_cnn.git
cd radiography_pulmonaires_cnn
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Un GPU n'est pas nécessaire : les résultats ont été obtenus sur CPU. Versions utilisées : Python **À COMPLÉTER**, PyTorch **À COMPLÉTER**, torchvision **À COMPLÉTER**.

## 8. Reproduire les résultats

Placer les données dans `dataset/Training` et `dataset/Testing`.

### A. Avec les notebooks (pour comprendre)
Lancer, depuis `notebooks/`, dans cet ordre : `model_resnet18.ipynb`, `model_cnn_scratch.ipynb`, `07_gradcam_errors.ipynb`.

### B. Avec les scripts (pour reproduire ou utiliser), depuis la racine du dépôt

```bash
# 1. Entraînement (10 époques pour resnet18, 30 pour cnn, lr 1e-4, comme dans les notebooks)
python src/train.py --model resnet18
python src/train.py --model cnn

# 2. Évaluation sur le test + sauvegarde des logits (utilisés pour choisir les images à examiner)
python src/evaluate.py --model resnet18 --save_logits results/logits_resnet18.npz
python src/evaluate.py --model cnn      --save_logits results/logits_cnn.npz

# 3. Grad-CAM
python src/gradcam.py groups                                                    # tailles des groupes d'erreurs
python src/gradcam.py overlay --model resnet18 --group resnet18_glioma_vers_notumor --n 6
python src/gradcam.py overlay --model resnet18 --indices 279 376 --true_class --signed   # carte signée de la VRAIE classe
python src/gradcam.py stats --model resnet18                                    # mesure sur les 1 600 images
python src/gradcam.py stats --model cnn
python src/gradcam.py summary                                                   # médianes par modèle / classe

# 4. Cadrage : entraînement contre test, par classe
python src/framing.py

# 5. Prédire sur une image (projet académique, pas un outil de diagnostic)
python src/predict.py --model resnet18 --image chemin/vers/irm.jpg --cam sortie.png
```

Sous Windows, ajouter `--num_workers 0` si le chargement des données signale une erreur de multiprocessing.

### Vérifier la reproductibilité sans réentraîner
- Avec les checkpoints du rapport, `python src/evaluate.py --model resnet18` doit afficher une exactitude test de **0,9431**, et `--model cnn` de **0,8675**.
- `python src/evaluate.py --model resnet18 --compare_logits results/logits_resnet18.npz` recalcule les logits et les compare à ceux du rapport (`=> REPRODUIT` si l'écart est inférieur à 1e-3).
- `python src/gradcam.py summary` se recalcule à partir des CSV `results/gradcam_stats_*.csv`, sans checkpoint.

### Ce qui n'est pas reproductible à l'identique
- Les **graines de PyTorch n'étaient pas fixées** pour les résultats du rapport : `train.py --seed 42` (par défaut) rend l'entraînement plus reproductible, mais ses scores seront proches, pas identiques, à ceux du rapport (la validation du CNN varie de ±3 points d'une époque à l'autre).
- `train.py` **refuse d'écraser** un checkpoint existant (`--overwrite` pour le remplacer, `--tag NOM` pour un nouvel essai) afin de protéger les checkpoints des résultats du rapport.
- Compiler le rapport : `cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## 9. Limites

- Une seule exécution par modèle ; budgets d'entraînement inégaux (10 contre 30 époques, poids pré-entraînés contre aléatoires). Le CNN n'avait pas convergé : son score est une borne basse.
- Pas d'identifiant de patient ; étiquettes du jeu non auditées ; un seul jeu de données, pas de validation externe.
- Grad-CAM : cartes 7×7 floues ; la ReLU masque les contributions négatives ; le masque de mesure couvre toute la tête (il distingue le fond, pas le cerveau) ; une carte n'établit pas de causalité.
- La mesure du cadrage mélange zoom, plan de coupe et effet du seuil d'intensité.
- Ce travail n'est pas un outil clinique.

## 10. Références

- M. Nickparvar, *Brain Tumor MRI Dataset*, Kaggle (lien en section 4).
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- B. Zhou et al., *Learning Deep Features for Discriminative Localization* (CAM), CVPR 2016.
- R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV 2017.
- J. Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018.
- R. Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence 2020.
- S. Ioffe, C. Szegedy, *Batch Normalization*, ICML 2015 ; D. Kingma, J. Ba, *Adam*, ICLR 2015 ; A. Paszke et al., *PyTorch*, NeurIPS 2019.

## 11. Auteur

RANARISON Rina — Master 2 Mathématiques appliquées (Calcul numérique), Université d'Antananarivo.

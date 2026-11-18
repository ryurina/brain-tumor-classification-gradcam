# Où regarde un CNN qui classe des IRM cérébrales ? Interprétation par Grad-CAM

## 1. Objectifs

1. Comparer les performances et les **erreurs** de deux modèles sur la classification d'IRM cérébrales en quatre classes (gliome, méningiome, pas de tumeur, tumeur de l'hypophyse).
2. **Expliquer Grad-CAM** : ses formules, leur justification, ce qu'une carte signifie et ne signifie pas.
3. **Interpréter** les cartes des deux modèles, sur des exemples et par une mesure sur toutes les images de test (part de la carte située sur le fond noir).

## 2. Résumé des résultats

### Performances (test, 1 600 images)

| | ResNet18 | CNN from scratch |
|---|---|---|
| Paramètres | ~11 millions | 393 844 |
| Entraînement | 10 époques, Adam, lr 1e-4 | 30 époques, Adam, lr 1e-4 |
| Meilleure exactitude de validation | 98,57 % | 92,23 % |
| **Exactitude sur le test** | **94,31 %** | **86,75 %** |
| Rappel du gliome | 0,82 | 0,695 |
| Tumeurs prédites « pas de tumeur » | 30 | 77 |

ResNet18 fait nettement mieux (écart de 7,6 points, bien au-delà de l'incertitude de ±1 à 2 points). Les deux modèles perdent 4 à 5 points entre validation et test, et le gliome est la classe la plus difficile.

<p align="center">
  <img src="results/figures/confusion_resnet18.png" width="48%" alt="Matrice de confusion de ResNet18 sur le test"/>
  <img src="results/figures/confusion_cnn.png" width="48%" alt="Matrice de confusion du CNN sur le test"/>
</p>

*Matrices de confusion sur le test (lignes : vraie classe). À gauche ResNet18, à droite le CNN.*

### Ce que montre Grad-CAM

| Mesure (médianes sur les 1 600 images) | ResNet18 | CNN |
|---|---|---|
| Part de la carte située sur le **fond noir** | 0,19 | 0,32 |
| Part de l'image occupée par le fond (référence : carte uniforme) | 0,42 | 0,42 |
| Idem pour les **gliomes** (carte / référence) | 0,24 / 0,49 | **0,49 / 0,49** |

1. **ResNet18 regarde la tête, pas le fond** : sa carte est sous la référence pour 99,9 % des images bien classées.
2. **Le CNN s'appuie davantage sur le fond**, surtout pour les gliomes, où il n'évite pas le fond plus que le hasard (carte supérieure à la référence pour 55,5 % des images). Cela rappelle un raccourci possible, sans le prouver.
3. **Les erreurs sont largement communes** : 77 des 91 erreurs de ResNet18 sont aussi des erreurs du CNN.
4. **Gliomes manqués par ResNet18** (4 exemples avec cartes signées) : la carte signée de « glioma » est **négative sur toute la tête, lésion comprise**. Aucune région ne soutient « glioma » : la lésion n'est pas reconnue comme gliome, et ce n'est pas un simple effet de « dilution » dans la moyenne. Pour « notumor », la lésion va **contre** cette classe dans 1 exemple sur 4 (image 97) et reste neutre ou rouge dans les autres.
5. **Piège de lecture** : sur la carte *non signée* de la vraie classe, des « points chauds » apparaissent parfois **hors de la tête** (image 243) alors qu'il n'y a aucune preuve positive : la normalisation image par image grossit des valeurs proches de zéro. Il faut lire cette carte avec la carte signée.

Ces observations sont **corrélationnelles** et portent sur peu d'exemples visuels : elles ne prouvent pas que les modèles utilisent le fond ou les bords.

<p align="center">
  <img src="results/figures/gradcam_comparaison.png" width="80%" alt="Comparaison Grad-CAM ResNet18 / CNN sur des erreurs communes"/>
</p>

*Mêmes images mal classées par les deux modèles : image, Grad-CAM de ResNet18, Grad-CAM du CNN (carte de la classe prédite ; titre rouge : erreur, avec la confiance entre parenthèses).*

## 3. Grad-CAM en bref

Pour une classe $c$ de score (logit) $y^c$ et la dernière couche convolutive, dont les $K$ cartes d'activation $A^k \in \mathbb{R}^{7\times 7}$ ($K=256$ pour le CNN, $512$ pour ResNet18) :

$$\alpha_k^c \;=\; \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}, \qquad Z = 7\times 7 = 49$$

$$L^c \;=\; \mathrm{ReLU}\!\Big(\sum_{k=1}^{K}\alpha_k^c\,A^k\Big)$$

La carte $L^c$ est agrandie à $224\times 224$ (interpolation bilinéaire) puis normalisée dans $[0,1]$.

- $\alpha_k^c$ est l'**importance du canal** $k$ pour la classe $c$ (moyenne spatiale du gradient) ; la somme pondérée des cartes dit **où** les canaux importants sont actifs.
- La **ReLU** ne garde que ce qui **soutient** la classe.
- **Pour nos deux modèles** (pooling moyen global puis couche linéaire de poids $W_{ck}$), $y^c = b_c + \sum_k W_{ck}\,\frac{1}{Z}\sum_{i,j}A^k_{ij}$, donc $\alpha_k^c = W_{ck}/Z$ : Grad-CAM coïncide avec **CAM** (Zhou et al., 2016). La carte est la contribution locale $s^c_{ij}=\sum_k W_{ck}A^k_{ij}$ de chaque position, et le score est leur moyenne :

$$y^c \;=\; b_c + \frac{1}{Z}\sum_{i,j} s^c_{ij}$$

- La **carte signée** supprime la ReLU et divise par $\max|L|$ (valeurs dans $[-1,1]$) : **rouge = pour la classe, bleu = contre**. Elle distingue une zone *sans effet* d'une zone qui *pousse contre* la classe, ce que la ReLU cache.

**Comment lire une carte** : elle est floue (cases de $32\times32$ pixels), normalisée image par image (il y a toujours un point « le plus chaud », même si la preuve est faible) et **n'établit pas la causalité**.

## 4. Données

- **Brain Tumor MRI Dataset** (Masoud Nickparvar, Kaggle) : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Classes : `glioma`, `meningioma`, `notumor`, `pituitary` (dossiers par classe).
- Copie utilisée : `Training` = 5 600 images, `Testing` = 1 600 images (400 par classe). La version exacte peut différer de celle décrite sur la page Kaggle.
- `Training` est séparé en entraînement (4 480 images) et validation (1 120 images) par tirage **stratifié** (`random_state=42`). `Testing` n'est utilisé qu'en évaluation finale.
- **Les données ne sont pas dans ce dépôt.** Téléchargez-les depuis Kaggle et placez-les dans `dataset/Training` et `dataset/Testing`.

## 5. Méthode

| | ResNet18 | CNN from scratch |
|---|---|---|
| Entrée | RGB, 224×224, normalisation ImageNet | niveaux de gris (1 canal), 224×224, moyenne 0,5 / écart-type 0,5 |
| Initialisation | poids ImageNet, fine-tuning complet | aléatoire |
| Architecture | ResNet18, dernière couche à 4 sorties | 5 blocs Conv-BatchNorm-ReLU-MaxPool (16→256 canaux), pooling global, dropout 0,3 |
| Augmentation (train seulement) | rotation ±10° | rotation ±10° |

- Perte : entropie croisée, lots de 32, sur CPU. Le checkpoint conservé est celui de meilleure exactitude de validation.
- **Grad-CAM** sur `layer4[-1]` (ResNet18) et `features[-1]` (CNN), classe prédite (ou vraie, avec `--true_class`).
- **Mesure sur le fond** : un masque du premier plan (seuillage, ouverture, remplissage ; il couvre toute la **tête**, pas seulement le cerveau) donne `off_cam` (part de la carte sur le fond) et `off_area` (part de l'image occupée par le fond : ce que donnerait une carte uniforme, la référence).

## 6. Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── notebooks/                          # la démarche, avec sorties et figures (à lire pour comprendre)
│   ├── model_resnet18.ipynb            # entraînement, évaluation, logits de ResNet18
│   ├── model_cnn_scratch.ipynb         # entraînement, évaluation, logits du CNN
│   └── gradcam_errors.ipynb         # Grad-CAM sur les erreurs, cartes signées, mesure du fond
├── src/                                # les mêmes étapes en scripts (à lancer pour reproduire / utiliser)
│   ├── utils.py        # noms des modèles et des fichiers, graines, appareil
│   ├── dataset.py      # transformations, découpage 80/20 (random_state=42), DataLoaders
│   ├── model.py        # SmallCNN, ResNet18 à 4 sorties, chargement des checkpoints
│   ├── train.py        # entraînement
│   ├── evaluate.py     # test : métriques, matrice de confusion, logits, vérification
│   ├── gradcam.py      # Grad-CAM, groupes d'erreurs, figures, cartes signées, mesure du fond, tableaux du rapport
│   └── predict.py      # prédiction sur une image (classe, confiance, Grad-CAM)
├── results/                            # logits (.npz), tableaux (.csv), figures
├── models/                             # checkpoints (.pth), NON versionnés
└── dataset/                            # données Kaggle, NON versionnées
```

Les **notebooks** sont la trace des résultats du rapport. Les **scripts** de `src/` reprennent le même code (mêmes transformations, mêmes architectures, mêmes hyperparamètres, même Grad-CAM) pour reproduire et réutiliser, sans notebook.

## 7. Installation

```bash
git clone https://github.com/ryurina/radiography_pulmonaires_cnn.git
cd radiography_pulmonaires_cnn
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Reproduire les résultats

Placer les données dans `dataset/Training` et `dataset/Testing`.

### A. Avec les notebooks (pour comprendre)
Lancer, depuis `notebooks/`, dans cet ordre : `model_resnet18.ipynb`, `model_cnn_scratch.ipynb`, `gradcam_errors.ipynb`.

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
python src/gradcam.py summary                                                   # médianes par modèle / classe + tableaux du rapport

# 4. Prédire sur une image
python src/predict.py --model resnet18 --image chemin/vers/irm.jpg --cam sortie.png
```

## 9. Limites

- Une seule exécution par modèle ; 10 contre 30 époques, poids pré-entraînés contre aléatoires. Le CNN n'avait pas convergé : son score est une borne basse.
- Grad-CAM : cartes 7×7 floues ; la ReLU masque les contributions négatives ; le masque de mesure couvre toute la tête (il distingue le fond, pas le cerveau) ; une carte n'établit pas de causalité.

## 10. Références

- M. Nickparvar, *Brain Tumor MRI Dataset*, Kaggle.
- R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV 2017.

## 11. Auteur

RANARISON Rina — Master 2 Mathématiques appliquées (Calcul numérique), Université d'Antananarivo.

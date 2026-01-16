# Leaf Classification — Projet 


L'objectif est de comparer plusieurs méthodes de classification supervisée sur le jeu de données Leaf Classification (Kaggle), en respectant une démarche scientifique rigoureuse : validation croisée, recherche d'hyperparamètres et évaluation sur un ensemble test indépendant.

---

## Équipe

- Imane Chadli
- Adien Gagnon
- Jacob Tremblay

---

## Objectifs du projet

L'objectif principal est d'implémenter au moins six classificateurs avec scikit-learn et de comparer leurs performances à l'aide de bonnes pratiques expérimentales. Nous utilisons la validation croisée stratifiée, la recherche d'hyperparamètres et des métriques adaptées au problème (log loss). Les performances finales sont évaluées sur un ensemble test indépendant.

---

## Jeu de données

Le jeu de données Leaf Classification provient de Kaggle et contient 990 observations d'entraînement réparties en 99 classes. Chaque observation est représentée par 193 caractéristiques numériques. Puisque les descripteurs sont déjà extraits, le besoin d'ingénierie de caractéristiques avancée est limité.

---

## Méthodologie

### Séparation des données

Les données sont divisées en deux ensembles distincts :

- 80% pour l'ensemble de développement (train_dev) utilisé pour la validation croisée et la recherche d'hyperparamètres
- 20% pour l'ensemble test final (holdout) utilisé une seule fois pour l'évaluation finale

La séparation est stratifiée par classe afin de préserver la distribution des données.

### Validation croisée

Nous utilisons une validation croisée stratifiée à 5 plis. La normalisation (StandardScaler) est apprise uniquement sur les données d'entraînement de chaque pli, puis appliquée au pli de validation correspondant. Cette approche garantit qu'il n'y a aucune fuite de données entre entraînement et test.

### Métriques utilisées

Trois métriques principales sont utilisées pour évaluer les modèles :

- Log loss (métrique principale, cohérente avec le challenge Kaggle)
- Accuracy
- Top-k accuracy (k = 5)

---

## Modèles implémentés

Six modèles ont été implémentés via une architecture orientée objet :

1. Régression logistique
2. k-plus proches voisins (KNN)
3. Forêt aléatoire (Random Forest)
4. SVM (Support Vector Machine)
5. Gradient Boosting
6. Réseau de neurones multi-couches (MLP)

Une fabrique de modèles permet une création uniforme et extensible des classificateurs.

---

## Recherche d'hyperparamètres

La recherche d'hyperparamètres a été effectuée pour le modèle MLP en utilisant RandomizedSearchCV. L'optimisation est basée sur la log loss et permet de comparer clairement le MLP baseline avec le MLP optimisé (modèle final).

---

## Résultats

Les résultats de chaque expérience sont sauvegardés dans le répertoire `results/runs/<date>__<nom_experience>/`. Les résultats finaux présentés dans le rapport se trouvent dans `results/runs/final_results`.

Fichiers importants :

- `comparaison.csv` : tableau récapitulatif (CV + holdout)
- `metriques.json` : métriques détaillées
- `results/figures/` : figures utilisées dans le rapport

Les figures comparent les performances en validation croisée versus ensemble test, en se concentrant sur la log loss et l'accuracy.

---

## Structure du projet

```
├── configs/                      # Fichiers YAML de configuration
├── data/                         # Données 
├── docs/                         # Documentation et diagrammes UML
├── notebooks/                    # Analyse exploratoire (Jupyter)
├── results/                      # Résultats expérimentaux
├── scripts/                      # Scripts d'exécution
├── src/leaf_classification/
│   ├── gestion_donnees/
│   ├── modelisation/
│   ├── optimisation_validation/
│   ├── gestion_experiences/
│   └── utils/
├── main.py                       # Point d'entrée du projet
├── requirements.txt
└── README.md
```

---

## Exécution du projet

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancer une expérience

```bash
python main.py --config configs/experience.yaml
```
### Diagramme de classes
Un diagramme UML décrivant l’architecture du projet est fourni dans :
```
docs/diagramme_classes.png

```
### Références
https://www.kaggle.com/c/leaf-classification

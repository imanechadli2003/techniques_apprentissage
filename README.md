# Leaf Classification — Projet IFT712

Projet de session du cours **IFT712 – Techniques d’apprentissage**.  
L’objectif est de comparer plusieurs méthodes de classification supervisée sur le jeu de données **Leaf Classification** (Kaggle), en respectant une démarche scientifique rigoureuse : validation croisée, recherche d’hyperparamètres et évaluation sur un ensemble test indépendant.

---

##  Équipe
- Imane Chadli
- Adien Gagnon
- Jacob Tremblay  

---

##  Objectifs du projet
- Implémenter **au moins six classificateurs** avec `scikit-learn`
- Comparer leurs performances à l’aide de **bonnes pratiques expérimentales**
- Utiliser :
  - validation croisée stratifiée
  - recherche d’hyperparamètres
  - métriques adaptées au problème (log loss)
- Évaluer les performances finales sur un **ensemble test indépendant**

---

##  Jeu de données
- **Leaf Classification** (Kaggle)
- 990 observations d’entraînement
- 99 classes
- 193 caractéristiques numériques par observation

Le jeu de données fournit déjà des descripteurs numériques, ce qui limite le besoin d’ingénierie de caractéristiques avancée.

---

##  Méthodologie

### Séparation des données
- **80 %** : ensemble de développement (`train_dev`)
  - validation croisée
  - recherche d’hyperparamètres
- **20 %** : ensemble test final (`holdout`)
  - utilisé une seule fois pour l’évaluation finale

La séparation est **stratifiée par classe** afin de préserver la distribution des données.

---

### Validation croisée
- Validation croisée **stratifiée à 5 plis**
- Normalisation (StandardScaler) :
  - apprise uniquement sur les données d’entraînement de chaque pli
  - appliquée ensuite au pli de validation
- Aucune fuite de données entre entraînement et test

---

### Métriques utilisées
- **Log loss** (métrique principale, cohérente avec le challenge Kaggle)
- Accuracy
- Top-k accuracy (k = 5)

---

##  Modèles implémentés
Les six modèles suivants ont été implémentés via une architecture orientée objet :

1. Régression logistique
2. k-plus proches voisins (KNN)
3. Forêt aléatoire (Random Forest)
4. SVM (Support Vector Machine)
5. Gradient Boosting
6. Réseau de neurones multi-couches (MLP)

Une **fabrique de modèles** permet une création uniforme et extensible des classificateurs.

---

##  Recherche d’hyperparamètres
- Recherche d’hyperparamètres effectuée pour le modèle MLP
- Méthode : `RandomizedSearchCV`
- Optimisation basée sur la **log loss**
- Comparaison claire entre :
  - MLP baseline
  - MLP tuné (modèle final)

---

##  Résultats
Les résultats de chaque expérience sont sauvegardés dans :
results/runs/<date>__<nom_experience>/
et nos résultats finals qu'on a présenté dans le rapport sont sauvegardées ici:
/results/runs/final_results

Fichiers importants :
- `comparaison.csv` : tableau récapitulatif (CV + holdout)
- `metriques.json` : métriques détaillées
- `results/figures/` : figures utilisées dans le rapport

Les figures comparent les performances :
- validation croisée vs ensemble test
- log loss et accuracy

---
##  Structure du projet
├── configs/ # Fichiers YAML de configuration
├── data/ # Données (raw, processed)
├── docs/ # Documentation et diagrammes UML
├── notebooks/ # Analyse exploratoire (Jupyter)
├── results/ # Résultats expérimentaux
├── scripts/ # Scripts d’exécution
├── src/leaf_classification/
│ ├── gestion_donnees/
│ ├── modelisation/
│ ├── optimisation_validation/
│ ├── gestion_experiences/
│ └── utils/
├── tests/ # Tests unitaires
├── main.py # Point d’entrée du projet
├── requirements.txt
└── README.md

##  Exécution du projet
### 1️ Installation des dépendances
```bash
pip install -r requirements.txt
### 2 Lancer une expérience:
```bash
python main.py --config configs/experience.yaml


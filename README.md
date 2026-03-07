# DEFT2013 Tâche 2 — Classification de recettes de cuisine

NOM Prenom - NOM Prenom

---

## Description de la tâche

L'objectif est de classifier automatiquement des recettes de cuisine en **3 catégories** : `Plat principal`, `Entrée` ou `Dessert`, à partir du titre, des ingrédients et des instructions de la recette.

**Exemple 1 — Plat principal** (`recette_221358.xml`)
> **Titre :** Feuilleté de saumon et de poireau, sauce aux crevettes
> **Ingrédients :** 1 gros pavé de saumon, 100g de crevettes décortiquées, 2 poireaux, 1 pâte feuilletée...
> **Label :** Plat principal

**Exemple 2 — Entrée** (`recette_48656.xml`)
> **Titre :** Cake poulet/moutarde/amandes
> **Ingrédients :** 3 œufs, 150g de farine, 1 sachet de levure, 100g de gruyère râpé...
> **Label :** Entrée

**Exemple 3 — Dessert** (`recette_71424.xml`)
> **Titre :** Gâteau au yaourt au coco sans huile de laetitia
> **Ingrédients :** 1 pot de yaourt, 1 pot de lait de coco, 3 oeufs, 3 pots de farine...
> **Label :** Dessert

---

## Statistiques corpus

### Taille des ensembles

| Ensemble  | Nombre de documents |
| --------- | ------------------: |
| Train     | 12 473              |
| Test      | 1 388               |
| **Total** | **13 861**          |

### Répartition des étiquettes

| Classe         | Train | % Train | Test | % Test |
| -------------- | ----: | ------: | ---: | -----: |
| Plat principal | 5 802 |  46,5%  |  644 |  46,4% |
| Dessert        | 3 762 |  30,2%  |  407 |  29,3% |
| Entrée         | 2 909 |  23,3%  |  337 |  24,3% |

> Le corpus est **déséquilibré** : la classe `Plat principal` représente presque la moitié des données. C'est pourquoi nous utilisons le **F1-score macro** comme métrique principale plutôt que l'accuracy.

### Colonnes disponibles

| Colonne       | Description                                               |
| ------------- | --------------------------------------------------------- |
| `doc_id`      | Identifiant unique de la recette                          |
| `titre`       | Titre de la recette (moy. 31 caractères)                  |
| `type`        | **Label cible** : Plat principal / Entrée / Dessert       |
| `difficulte`  | Très facile / Facile / Moyennement difficile / Difficile  |
| `cout`        | Bon marché / Moyen / Assez Cher                           |
| `ingredients` | Liste des ingrédients (moy. 214 caractères)               |
| `recette`     | Instructions de préparation (moy. 723 caractères)         |

---

## Installation
```bash
git clone https://github.com/11Maroua/NLP_project.git
cd NLP_project
pip install -r requirements.txt
```

Placer les fichiers `train.csv` et `test.csv` dans le dossier `data/`.

---

## Méthodes proposées

### Run 1 : Baseline — Classe majoritaire

**Descripteurs utilisés :** aucun (le texte n'est pas lu)

**Classifieur :** `DummyClassifier(strategy='most_frequent')` — prédit toujours `Plat principal`

**Principe :** Sans aucun traitement NLP, la stratégie naïve optimale est de prédire systématiquement la classe la plus représentée dans le train (46,5%). Sert de borne inférieure pour évaluer l'apport réel de nos méthodes.

**Résultats :**

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.00 |   0.00 | 0.00 |     407 |
| Entrée         |      0.00 |   0.00 | 0.00 |     337 |
| Plat principal |      0.46 |   1.00 | 0.63 |     644 |
| **macro avg**  |  **0.15** | **0.33** | **0.211** | **1388** |

---

### Run 2 : TF-IDF + SVM

**Descripteurs utilisés :** représentation TF-IDF sur la concaténation `titre + ingrédients + recette`

**Pré-traitement :**
- Mise en minuscules
- Suppression des caractères spéciaux et chiffres
- Suppression des stopwords français (NLTK)
- Stemming (SnowballStemmer français)

**Vectorisation :** `TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)`

**Classifieur :** `SVC(kernel='linear', C=1.0)`

**Principe :** Représentation bag-of-words pondérée par la rareté des termes dans le corpus. Nous avons testé 3 combinaisons de colonnes :

| Features                          | F1 macro |
| --------------------------------- | -------: |
| Titre seul                        |    0.816 |
| Titre + ingrédients               |    0.825 |
| **Titre + ingrédients + recette** | **0.862** |

**Résultats détaillés (meilleure config — titre + ingrédients + recette) :**

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.99 |   1.00 | 0.99 |     407 |
| Entrée         |      0.77 |   0.69 | 0.72 |     337 |
| Plat principal |      0.85 |   0.89 | 0.87 |     644 |
| **macro avg**  |  **0.87** | **0.86** | **0.862** | **1388** |

---

### Run 3 : Embeddings statiques vs dynamiques

#### Run 3a : Word2Vec + SVM — Embeddings statiques
**Modèle :** `Word2Vec(vector_size=100, window=5, min_count=2, epochs=10)`
**Résultats :** `à compléter après Colab`

#### Run 3b : CamemBERT + SVM — Embeddings dynamiques
**Modèle :** `camembert-base` via HuggingFace Transformers
**Résultats :** `à compléter après Colab`

---

### Run 4 : CamemBERT + TF-IDF + SVM (pour aller plus loin)
**Résultats :** `à compléter après Colab`

---

## Résultats

| Run    | Méthode                                | F1 macro      |
| ------ | -------------------------------------- | ------------: |
| Run 1  | Baseline (classe majoritaire)          | 0.211         |
| Run 2a | TF-IDF + SVM (titre seul)              | 0.816         |
| Run 2b | TF-IDF + SVM (titre + ingr.)           | 0.825         |
| Run 2c | TF-IDF + SVM (titre + ingr. + recette) | **0.862**     |
| Run 3a | Word2Vec + SVM                         | `à compléter` |
| Run 3b | CamemBERT + SVM                        | `à compléter` |
| Run 4  | CamemBERT + TF-IDF + SVM              | `à compléter` |

---

## Analyse des résultats

### Observations Run 2
- **Dessert** est la classe la mieux classifiée (F1 = 0.99)
- **Entrée** est la plus difficile (F1 = 0.72) — confusion avec `Plat principal`
- Ajouter `recette` améliore le F1 de +0.037 par rapport au titre seul

### Matrice de confusion
*(à compléter)*

### Mots les plus décisifs par classe
*(à compléter)*

---

## Réflexion critique

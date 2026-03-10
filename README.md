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

**Principe :** Sert de borne inférieure pour évaluer l'apport réel de nos méthodes.

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

**Optimisation :** Grid Search 5-folds sur 27 combinaisons d'hyperparamètres

**Meilleurs paramètres trouvés et réutilisés plus tard dans d'autres runs :**
- `ngram_range = (1,2)` — unigrammes + bigrammes
- `max_features = 10 000`
- `C = 1.0`

**Classifieur :** `SVC(kernel='linear', C=1.0)`

**Stabilité (cross-validation 5-folds) :**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Moyenne | Écart-type |
| -----: | -----: | -----: | -----: | -----: | ------: | ---------: |
|  0.860 |  0.858 |  0.867 |  0.858 |  0.868 |   0.862 |  **±0.004** |

> Écart-type très faible (±0.004) → modèle ** stable**, on sait que les scores ne dépendent pas du hasard.

**Résultats sur le jeu de test :**

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.99 |   1.00 | 0.99 |     407 |
| Entrée         |      0.76 |   0.70 | 0.73 |     337 |
| Plat principal |      0.85 |   0.88 | 0.87 |     644 |
| **macro avg**  |  **0.87** | **0.86** | **0.863** | **1388** |

> F1 CV (0.862) ≈ F1 test (0.863) → **pas de surapprentissage**, le modèle généralise bien.

---

### Run 3 : Embeddings statiques vs dynamiques

#### Run 3a : Word2Vec + SVM — Embeddings statiques

**Descripteurs :
**Modèle :** `Word2Vec(vector_size=100, window=5, min_count=2, epochs=10)` — vocab : 11 866 mots
**Vectorisation :** moyenne des vecteurs Word2Vec sur `titre + ingrédients + recette`

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.98 |   0.99 | 0.99 |     407 |
| Entrée         |      0.75 |   0.64 | 0.69 |     337 |
| Plat principal |      0.83 |   0.89 | 0.86 |     644 |
| **macro avg**  |  **0.85** | **0.84** | **0.844** | **1388** |


#### Run 3b : CamemBERT + SVM — Embeddings dynamiques

**Descripteurs :** embeddings contextuels CamemBERT (`camembert-base`), moyenne du dernier layer
**Vectorisation :** moyenne du dernier layer sur `titre + ingrédients + recette`

**Principe :** Contrairement à Word2Vec (vecteur fixe par mot), CamemBERT produit des représentations **contextuelles** — le même mot aura un vecteur différent selon son contexte. Modèle pré-entraîné sur 138GB de texte français.

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.97 |   1.00 | 0.98 |     407 |
| Entrée         |      0.75 |   0.59 | 0.66 |     337 |
| Plat principal |      0.82 |   0.89 | 0.85 |     644 |
| **macro avg**  |  **0.85** | **0.83** | **0.833** | **1388** |
---

### Run 4 : Pour aller plus loin

Trois variantes testées pour dépasser le TF-IDF seul.

#### Run 4a : TF-IDF + CamemBERT + SVM

**Descripteurs :** TF-IDF (10 000 dim.) + embeddings CamemBERT (768 dim.) = 10 768 features, combinés via `scipy.sparse.hstack`

**Principe :** CamemBERT apporte la sémantique contextuelle, TF-IDF apporte la précision lexicale sur le vocabulaire culinaire spécifique.

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.99 |   1.00 | 0.99 |     407 |
| Entrée         |      0.78 |   0.71 | 0.74 |     337 |
| Plat principal |      0.86 |   0.90 | 0.88 |     644 |
| **macro avg**  |  **0.88** | **0.87** | **0.873** | **1388** |

#### Run 4b : TF-IDF + Word2Vec + SVM

**Descripteurs :** TF-IDF (10 000 dim.) + embeddings Word2Vec (100 dim.) = 10 100 features, combinés via `scipy.sparse.hstack`

**Principe :** Word2Vec entraîné sur le corpus culinaire capture la sémantique du domaine, TF-IDF apporte la précision lexicale.

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.99 |   1.00 | 0.99 |     407 |
| Entrée         |      0.77 |   0.70 | 0.73 |     337 |
| Plat principal |      0.85 |   0.89 | 0.87 |     644 |
| **macro avg**  |  **0.87** | **0.86** | **0.865** | **1388** |

#### Run 5 : CamemBERT fine-tuné

**Principe :** Fine-tuning complet de `camembert-base` (110M paramètres) directement sur la tâche de classification. Contrairement aux runs précédents qui utilisent CamemBERT comme extracteur de features figé, ici tous les paramètres sont mis à jour pendant l'entraînement.

**Hyperparamètres :**
- `learning_rate = 1e-5`
- `batch_size = 16` + `gradient_accumulation_steps = 2`
- `num_train_epochs = 5` avec early stopping (patience=3)
- `fp16 = True` (mixed precision GPU)
- Texte : `titre + ingrédients + recette[:200]`, `max_length = 256`

**Résultats :** F1 macro = **0.879**

---

## Résultats

| Run        | Méthode                                |  F1 macro |
| ---------- | -------------------------------------- | --------: |
| Run 1      | Baseline (classe majoritaire)          |     0.211 |
| Run 2a     | TF-IDF + SVM (titre seul)              |     0.816 |
| Run 2b     | TF-IDF + SVM (titre + ingr.)           |     0.825 |
| Run 2c     | TF-IDF + SVM (titre + ingr. + recette) |     0.863 |
| Run 3a     | Word2Vec + SVM                         |     0.844 |
| Run 3b     | CamemBERT + SVM                        |     0.833 |
| Run 4b     | TF-IDF + Word2Vec + SVM                |     0.865 |
| Run 4a     | TF-IDF + CamemBERT + SVM               |     0.873 |
| **Run 4c** | **CamemBERT fine-tuné**                | **0.879** |

---

## Analyse des résultats

### Observations générales

- La progression est claire : Baseline (0.211) → TF-IDF (0.863) → Fine-tuning (0.879).
- **Dessert** est la classe la mieux classifiée dans tous les runs — *sucre*, *chocolat*, *farine* sont des marqueurs quasi exclusifs.
- **Entrée** est systématiquement la plus difficile — forte confusion avec `Plat principal` car les deux partagent un vocabulaire culinaire similaire.

### Apport des embeddings

- Word2Vec seul (0.844) ≈ TF-IDF seul (0.863) — les embeddings statiques capturent bien la sémantique du domaine.
- CamemBERT seul (0.833) mais combiné au TF-IDF (0.873) il devient très compétitif.
- La combinaison TF-IDF + embeddings améliore toujours par rapport aux méthodes seules 

### Apport du fine-tuning

- Le fine-tuning CamemBERT (0.879) est la **meilleure méthode** — adapter tous les paramètres du modèle à la tâche surpasse les embeddings figés.
- Le gain par rapport au TF-IDF seul est de +0.016, ce qui confirme l'apport des représentations contextuelles profondes bien que ce soit une amélioration assez limité pour une approche bien plus complexe que le simple TF-IDF.
- La classe `Entrée` progresse à chaque run plus sophistiqué : 0.69 (Word2Vec) → 0.73 (TF-IDF+Word2Vec) → 0.74 (TF-IDF+CamemBERT) → 0.879 globalement (fine-tuning).

### Apport de la cross-validation (Run 2)

- F1 CV (0.862 ± 0.004) ≈ F1 test (0.863) — faible écart-type et cohérence train/test confirment l'absence de surapprentissage.

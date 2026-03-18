# Classification de recettes de cuisine

ABID Ikram - NAIT SLIMANI Marouan

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

### Questions d'exploration

**Les titres seuls sont-ils discriminants ?**
Oui partiellement — le TF-IDF sur titre seul atteint F1=0.816, ce qui est déjà solide. Des titres comme *Gâteau au chocolat* ou *Carpaccio de bœuf* sont très discriminants. Mais beaucoup de titres sont ambigus : *Cake poulet/moutarde* peut être une entrée ou un plat selon le contexte.

**Les instructions seules suffisent-elles ?**
Le texte complet `titre + ingrédients + recette` (F1=0.863) surpasse largement le titre seul (F1=0.816), ce qui confirme que les instructions apportent un signal supplémentaire utile — notamment pour distinguer `Entrée` de `Plat principal`.

**Certaines recettes semblent-elles ambiguës ?**
Oui, notamment entre `Entrée` et `Plat principal`. Une quiche, un gratin individuel ou une salade composée peuvent être servis en entrée ou en plat selon les portions et le contexte du repas. Cette ambiguïté est intrinsèque aux données et constitue la principale limite de nos modèles.

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
- Mise en minuscules, suppression des caractères spéciaux et chiffres
- Suppression des stopwords français (NLTK)
- Stemming (SnowballStemmer français)

**Optimisation :** Grid Search 5-folds sur 27 combinaisons d'hyperparamètres — meilleurs paramètres trouvés et réutilisés dans les runs suivants :
- `ngram_range = (1,2)` — unigrammes + bigrammes
- `max_features = 10 000`
- `C = 1.0`

**Classifieur :** `SVC(kernel='linear', C=1.0, random_state=42)`

**Stabilité (cross-validation 5-folds) :**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Moyenne | Écart-type |
| -----: | -----: | -----: | -----: | -----: | ------: | ---------: |
|  0.860 |  0.858 |  0.867 |  0.858 |  0.868 |   0.862 |  **±0.004** |

> Écart-type très faible (±0.004) → modèle très stable, les scores ne dépendent pas du hasard.

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.99 |   1.00 | 0.99 |     407 |
| Entrée         |      0.76 |   0.70 | 0.73 |     337 |
| Plat principal |      0.85 |   0.88 | 0.87 |     644 |
| **macro avg**  |  **0.87** | **0.86** | **0.863** | **1388** |

> F1 CV (0.862) ≈ F1 test (0.863) → pas de surapprentissage, le modèle généralise bien.

---

### Run 3 : Embeddings statiques vs dynamiques

#### Run 3a : Word2Vec + SVM

**Descripteurs :** vecteur moyen Word2Vec (100 dimensions) entraîné sur le corpus de recettes
**Modèle :** `Word2Vec(vector_size=100, window=5, min_count=2, epochs=10, seed=42)` — vocab : 11 866 mots
**Vectorisation :** moyenne des vecteurs Word2Vec sur `titre + ingrédients + recette`

**Principe :** Chaque recette est représentée par la moyenne des vecteurs Word2Vec de ses tokens. Contrairement au TF-IDF, des mots sémantiquement proches (*poulet*, *volaille*) auront des vecteurs proches dans l'espace d'embedding.

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.98 |   0.99 | 0.99 |     407 |
| Entrée         |      0.75 |   0.64 | 0.69 |     337 |
| Plat principal |      0.83 |   0.89 | 0.86 |     644 |
| **macro avg**  |  **0.85** | **0.84** | **0.844** | **1388** |

#### Run 3b : CamemBERT + SVM

**Descripteurs :** embeddings contextuels CamemBERT (`camembert-base`), moyenne du dernier layer
**Vectorisation :** sur `titre + ingrédients + recette`, `max_length=256`

**Principe :** Contrairement à Word2Vec (vecteur fixe par mot), CamemBERT produit des représentations **contextuelles** — le même mot aura un vecteur différent selon son contexte. Modèle pré-entraîné sur 138GB de texte français.

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.97 |   1.00 | 0.98 |     407 |
| Entrée         |      0.75 |   0.59 | 0.66 |     337 |
| Plat principal |      0.82 |   0.89 | 0.85 |     644 |
| **macro avg**  |  **0.85** | **0.83** | **0.833** | **1388** |

---

### Run 4 : Combinaison TF-IDF + Embeddings + SVM

Deux variantes pour combiner la précision lexicale du TF-IDF avec la richesse sémantique des embeddings via `scipy.sparse.hstack`.

#### Run 4a : TF-IDF + CamemBERT + SVM

**Descripteurs :** TF-IDF (10 000 dim.) + embeddings CamemBERT (768 dim.) = 10 768 features

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.99 |   1.00 | 0.99 |     407 |
| Entrée         |      0.78 |   0.71 | 0.74 |     337 |
| Plat principal |      0.86 |   0.90 | 0.88 |     644 |
| **macro avg**  |  **0.88** | **0.87** | **0.873** | **1388** |

#### Run 4b : TF-IDF + Word2Vec + SVM

**Descripteurs :** TF-IDF (10 000 dim.) + embeddings Word2Vec (100 dim.) = 10 100 features

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.99 |   1.00 | 0.99 |     407 |
| Entrée         |      0.77 |   0.70 | 0.73 |     337 |
| Plat principal |      0.85 |   0.89 | 0.87 |     644 |
| **macro avg**  |  **0.87** | **0.86** | **0.865** | **1388** |

---

### Run 5 : CamemBERT fine-tuné (pour aller plus loin)

**Principe :** Fine-tuning complet de `camembert-base` (110M paramètres) directement sur la tâche de classification. Contrairement aux runs précédents qui utilisent CamemBERT comme extracteur de features figé, ici tous les paramètres sont mis à jour pendant l'entraînement sur nos données.

**Hyperparamètres :**
- `learning_rate = 1e-5`, `batch_size = 16`, `gradient_accumulation_steps = 2`
- `num_train_epochs = 5` avec early stopping (patience=3)
- `fp16 = True` (mixed precision GPU T4)
- Texte : `titre + ingrédients + recette[:200]`, `max_length = 256`

| Classe         | Précision | Rappel |  F1  | Support |
| -------------- | --------: | -----: | ---: | ------: |
| Dessert        |      0.99 |   1.00 | 0.99 |     407 |
| Entrée         |      0.79 |   0.73 | 0.76 |     337 |
| Plat principal |      0.87 |   0.91 | 0.89 |     644 |
| **macro avg**  |  **0.88** | **0.88** | **0.879** | **1388** |

---

### Exploration : SMOTE pour améliorer la classe Entrée

Face aux difficultés persistantes sur la classe `Entrée`, nous avons testé **SMOTE** (Synthetic Minority Oversampling TEchnique) : au lieu de dupliquer des exemples, SMOTE génère de nouveaux exemples synthétiques en interpolant entre les exemples existants de la classe minoritaire dans l'espace TF-IDF.

**Principe :** Rééquilibrer les trois classes à 5 802 exemples chacune avant l'entraînement du SVM.

| Classe         | Baseline (sans SMOTE) | Avec SMOTE | Variation |
| -------------- | --------------------: | ---------: | --------: |
| F1 Dessert     |                  0.99 |       0.99 |         = |
| F1 Entrée      |                  0.73 |   **0.74** |     +0.01 |
| Recall Entrée  |                  0.70 |   **0.77** |     +0.07 |
| F1 Plat princ. |                  0.87 |       0.86 |     -0.01 |
| **F1 macro**   |              **0.863** |   **0.865** |   +0.002 |

SMOTE améliore le recall d'`Entrée` (+0.07) mais au prix d'une légère baisse sur `Plat principal` — trade-off classique du rééchantillonnage. Le gain global reste modeste (+0.002), confirmant que le problème est **intrinsèque** au vocabulaire partagé entre les deux classes.

---

## Résultats

| Run        | Méthode                                | F1 macro  |
| ---------- | -------------------------------------- | --------: |
| Run 1      | Baseline (classe majoritaire)          |     0.211 |
| Run 2a     | TF-IDF + SVM (titre seul)              |     0.816 |
| Run 2b     | TF-IDF + SVM (titre + ingr.)           |     0.825 |
| Run 2c     | TF-IDF + SVM (titre + ingr. + recette) |     0.863 |
| Run 2c+    | TF-IDF + SVM + SMOTE                   |     0.865 |
| Run 3a     | Word2Vec + SVM                         |     0.844 |
| Run 3b     | CamemBERT + SVM                        |     0.833 |
| Run 4b     | TF-IDF + Word2Vec + SVM                |     0.865 |
| Run 4a     | TF-IDF + CamemBERT + SVM               |     0.873 |
| **Run 5**  | **CamemBERT fine-tuné**                | **0.879** |

---

## Analyse des résultats

### Analyse quantitative

**Convergence des scores :** Les résultats convergent entre 0.84 et 0.88 pour toutes les méthodes non-triviales. Le plafond semble se situer autour de 0.88–0.90, ce qui suggère une limite liée aux données elles-mêmes plutôt qu'aux modèles.

**Impact du déséquilibre :** La classe `Entrée` (23% du corpus) obtient systématiquement le F1 le plus bas dans tous les runs. Le SMOTE améliore légèrement son recall (0.70 → 0.77) mais ne résout pas le problème fondamental.

**Progression par méthode :**
- Baseline → TF-IDF : +0.652 (apport massif du traitement textuel)
- TF-IDF → Combinaison TF-IDF+CamemBERT : +0.010 (apport des embeddings contextuels)
- Combinaison → Fine-tuning : +0.006 (apport de l'adaptation du modèle)

### Analyse qualitative

**Exemples de bonnes prédictions :**
- *"Moelleux au chocolat"* → Dessert ✓ (vocabulaire très discriminant)
- *"Bœuf bourguignon"* → Plat principal ✓ (terme culinaire spécifique)
- *"Verrine de saumon fumé"* → Entrée ✓ (le terme "verrine" est très discriminant)

**Exemples d'erreurs typiques :**
- *"Quiche lorraine"* → prédit Plat principal, label Entrée (peut être les deux selon le contexte)
- *"Cake salé au jambon"* → prédit Entrée, label Plat principal (ambiguïté réelle)
- *"Velouté de poireaux"* → prédit Entrée, label Plat principal (soupe légère = plat ou entrée ?)

**Mots les plus discriminants (SVM linéaire, Run 2) :**
- **Dessert :** *chocolat, sucr, gateau, vanill, farin, beurr, sucr glac, crème pâtissière*
- **Entrée :** *verrin, terrin, toast, velouté, carpaccio, mise bouche, feuilleté*
- **Plat principal :** *gratin, tajin, blanquett, bourguignon, lasagn, mijot, rôti, cassoulet*

### Pourquoi la confusion Entrée/Plat persiste-t-elle ?

La frontière entre `Entrée` et `Plat principal` est **intrinsèquement floue** dans les données pour plusieurs raisons :

1. **Vocabulaire partagé :** *poulet, légumes, sauce, fromage* apparaissent dans les deux classes avec des fréquences similaires — aucun modèle ne peut discriminer sur la base du vocabulaire seul.

2. **Ambiguïté de l'annotation :** Le label dépend du jugement subjectif de l'auteur de la recette, pas d'une règle objective. Une même recette pourrait légitimement être étiquetée différemment selon les auteurs.

3. **Limites du SMOTE :** Générer des exemples synthétiques d'`Entrée` dans l'espace TF-IDF ne crée pas de nouveau signal discriminant — les vecteurs synthétiques restent dans la même zone de confusion avec `Plat principal`.

4. **Limite du fine-tuning :** Même CamemBERT fine-tuné (F1 Entrée = 0.76) ne résout pas complètement le problème, ce qui confirme que la confusion est liée aux données et non aux modèles.

---

## Réflexion critique

**La tâche est-elle bien définie ?**
Partiellement. La distinction `Dessert` vs le reste est très bien définie et tous nos modèles l'identifient parfaitement (F1 ≥ 0.98). En revanche, la frontière `Entrée` / `Plat principal` repose sur un jugement contextuel (taille des portions, position dans le repas) qui n'est pas capturé par le texte seul. La tâche est donc bien définie pour deux classes sur trois.

**Une recette peut-elle appartenir à plusieurs catégories ?**
Oui, et c'est le problème central. Un velouté de légumes, une quiche ou un cake salé peuvent légitimement être une entrée ou un plat selon le contexte. Le schéma d'annotation mono-label force un choix arbitraire qui introduit du bruit dans les données et crée une frontière de décision artificielle que les modèles peinent à apprendre.

**Les classes sont-elles naturellement séparables ?**
`Dessert` l'est clairement (F1 = 0.99 dans tous les runs) grâce à un vocabulaire spécifique et exclusif. `Entrée` et `Plat principal` ne le sont pas naturellement — leurs distributions lexicales se chevauchent significativement. Nos expériences (SMOTE, classification hiérarchique, embeddings) confirment que ce chevauchement est une propriété des données et non un artefact de nos méthodes.

**La macro-F1 est-elle la meilleure métrique ?**
Oui pour ce corpus. Avec 46% de `Plat principal`, l'accuracy serait trompeuse — un modèle prédisant toujours `Plat principal` atteindrait 46% d'accuracy (notre baseline). La macro-F1 pénalise également les mauvaises performances sur `Entrée` (classe minoritaire), ce qui force les modèles à la traiter sérieusement.

**Vos modèles généralisent-ils réellement ?**
Pour le Run 2 : F1 CV (0.862 ± 0.004) ≈ F1 test (0.863) — l'absence de surapprentissage est confirmée. Pour le fine-tuning (Run 5), le split train/validation stratifié (90/10) et l'early stopping garantissent une bonne généralisation. Cependant, tous nos modèles sont entraînés sur un seul site web de recettes — leurs performances pourraient être moindres sur des recettes d'autres sources avec des styles d'écriture différents.
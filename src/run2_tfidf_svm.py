import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from utils import load_data, preprocess, build_text

# ── Chargement ──────────────────────────────────────────────────
train, test = load_data()
y_train = train['type']
y_test  = test['type']

# ── Texte : meilleure config trouvée = titre + ingredients + recette
X_train_text = build_text(train, ['titre', 'ingredients', 'recette']).apply(preprocess)
X_test_text  = build_text(test,  ['titre', 'ingredients', 'recette']).apply(preprocess)

# ── Pipeline TF-IDF + SVM ───────────────────────────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True)),
    ('clf',   SVC(kernel='linear'))
])

# ════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Grid Search pour trouver les meilleurs hyperparamètres
# ════════════════════════════════════════════════════════════════
print("="*60)
print("ÉTAPE 1 — Grid Search ...")
print("="*60)

param_grid = {
    'tfidf__ngram_range'  : [(1,1), (1,2), (1,3)],
    'tfidf__max_features' : [5000, 10000, 15000],
    'clf__C'              : [0.1, 1.0, 10.0],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,                    # 5-folds cross-validation
    scoring='f1_macro',
    n_jobs=-1,              
    verbose=1
)
grid.fit(X_train_text, y_train)

print(f"\nMeilleurs paramètres : {grid.best_params_}")
print(f"Meilleur F1 macro CV : {grid.best_score_:.3f}")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Cross-validation avec les meilleurs paramètres
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ÉTAPE 2 — Cross-validation 5-folds")
print("="*60)

best_pipeline = grid.best_estimator_
cv_scores = cross_val_score(
    best_pipeline,
    X_train_text,
    y_train,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

print(f"\nScores par fold : {[round(s,3) for s in cv_scores]}")
print(f"F1 macro moyen  : {cv_scores.mean():.3f}")
print(f"Écart-type      : {cv_scores.std():.3f}")

if cv_scores.std() < 0.01:
    print("→ Modèle très stable")
elif cv_scores.std() < 0.03:
    print("→ Modèle stable")
else:
    print("→ Modèle instable")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Évaluation finale sur le test avec les meilleurs params
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ÉTAPE 3 — Évaluation finale sur le jeu de test")
print("="*60)

best_pipeline.fit(X_train_text, y_train)
y_pred = best_pipeline.predict(X_test_text)

print(classification_report(y_test, y_pred))
print(f"F1 macro test : {f1_score(y_test, y_pred, average='macro'):.3f}")

# ════════════════════════════════════════════════════════════════
# RÉCAP FINAL
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RÉCAP")
print("="*60)
print(f"Meilleurs paramètres  : {grid.best_params_}")
print(f"F1 macro CV (train)   : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"F1 macro test         : {f1_score(y_test, y_pred, average='macro'):.3f}")
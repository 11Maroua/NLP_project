import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from utils import load_data, preprocess, build_text

# ── Chargement de nos données train-test ──────────────────────────────────────────────────
train, test = load_data()
y_train = train['type']
y_test  = test['type']

# ── Experimentation : on teste plusieurs combinaisons de colonnes ─────
combinaison_features = {
    'titre seul'                    : ['titre'],
    'titre + ingredients'           : ['titre', 'ingredients'],
    'titre + ingredients + recette' : ['titre', 'ingredients', 'recette'],
}

resultats = {}

for nom, colonnes in combinaison_features.items():
    print(f"\n{'='*50}")
    print(f"Features Selectionnées : {nom}")
    print('='*50)

    # Construire et prétraiter le texte
    X_train_text = build_text(train, colonnes).apply(preprocess)
    X_test_text  = build_text(test,  colonnes).apply(preprocess)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec  = vectorizer.transform(X_test_text)

    # SVM
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    # Résultats
    f1 = f1_score(y_test, y_pred, average='macro')
    resultats[nom] = round(f1, 3)
    print(classification_report(y_test, y_pred))
    print(f"F1 macro : {f1:.3f}")

# ── Tableau récapitulatif ────────────────────────────────────────
print("\n=== RÉCAP TF-IDF + SVM ===")
for nom, score in resultats.items():
    print(f"{nom:<40} F1 macro = {score}")
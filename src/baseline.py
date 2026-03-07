import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report

# ── Chargement des données ──────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

X_train = train['titre']
y_train = train['type']
X_test  = test['titre']
y_test  = test['type']

# ── Baseline : toujours prédire la classe majoritaire ───────────
clf = DummyClassifier(strategy='most_frequent')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ── Résultats ───────────────────────────────────────────────────
print("=== BASELINE — Classe majoritaire ===")
print(classification_report(y_test, y_pred))
print(f"F1 macro : {f1_score(y_test, y_pred, average='macro'):.3f}")
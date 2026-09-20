"""
UrbanCart Milestone 2 - predicting customer churn.

Problem type: supervised CLASSIFICATION on an imbalanced target (~7.8% churn).
Accuracy is deliberately not used to select a model; see the notebook for why.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
)

RANDOM_STATE = 42

# ---------- 1. Load ----------
CANDIDATES = [
    "data/milestone-2-customer-churn.csv",
    "milestone-2-customer-churn.csv",
    "/mnt/user-data/uploads/milestone-2-customer-churn.csv",
]
DATA_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("Place milestone-2-customer-churn.csv in ./data/")

df = pd.read_csv(DATA_PATH)
X, y = df.drop(columns="Churned"), df["Churned"]
print(f"{len(df):,} customers | churn rate {y.mean()*100:.2f}%")

# ---------- 2. Stratified split, before any training ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Train {len(X_train):,} | Test {len(X_test):,} ({y_test.sum()} churners held out)\n")

# ---------- 3. Why accuracy is the wrong yardstick ----------
lazy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
lazy_pred = lazy.predict(X_test)
_, _, lazy_fn, _ = confusion_matrix(y_test, lazy_pred).ravel()
print("Baseline 'nobody churns':")
print(f"  accuracy {accuracy_score(y_test, lazy_pred)*100:.1f}% | churners caught 0 | missed {lazy_fn}\n")

# ---------- 4. Candidates ----------
candidates = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]),
    "Random Forest": Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                         random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scale", StandardScaler()),
        ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]),
}

# ---------- 5. Tune the decision threshold on TRAINING folds only ----------
# Missing a churner costs far more than a false alarm, so optimise F2 (recall-weighted).
grid = np.arange(0.02, 0.91, 0.01)
rows = {}

for name, pipe in candidates.items():
    oof = cross_val_predict(pipe, X_train, y_train, cv=5, method="predict_proba")[:, 1]
    f2 = [fbeta_score(y_train, (oof >= t).astype(int), beta=2, zero_division=0) for t in grid]
    threshold = grid[int(np.argmax(f2))]

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    rows[name] = {
        "Model": name,
        "Threshold": threshold,
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "F2": fbeta_score(y_test, pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }

results = pd.DataFrame(rows.values()).sort_values("F2", ascending=False)
print("=== Tuned thresholds, scored on the held-out test set ===")
print(results.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

# ---------- 6. Chosen model in operational terms ----------
best = results.iloc[0]
flagged = best["TP"] + best["FP"]
print(f"\n=== {best['Model']} @ threshold {best['Threshold']:.2f} ===")
print(f"Flagged for outreach: {flagged:,} of {len(y_test):,} customers")
print(f"Churners caught:      {best['TP']} of {best['TP'] + best['FN']} ({best['Recall']*100:.0f}%)")
print(f"Precision:            {best['Precision']:.3f} vs {y_test.mean():.3f} at random "
      f"({best['Precision']/y_test.mean():.1f}x lift)")
print(f"Contacts per churner: {flagged/best['TP']:.1f}")

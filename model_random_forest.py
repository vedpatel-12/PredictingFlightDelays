"""
Random Forest Model for Predicting Flight Delay Rate
------------------------------------------------------
A Random Forest trains many decision trees, each on a random bootstrap sample
of the data and a random subset of features at each split. Predictions are
averaged across all trees (bagging). Because each tree sees different data and
features, the ensemble is much more robust than a single tree and naturally
handles non-linear relationships (e.g., delay spikes only in certain months
for certain carriers).

Strengths: handles non-linearity, robust to outliers, built-in feature
importance, no need to scale features.
Weakness: slower to train than linear models, less interpretable per-tree.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("airline_delay_cleaned.csv")

# ── Feature selection (non-leaky columns only) ─────────────────────────────────
features = ["month", "carrier", "airport", "arr_flights",
            "avg_airline_delay_rate", "avg_airport_delay_rate"]
target = "delay_rate"

X = pd.get_dummies(df[features], columns=["carrier", "airport"])
y = df[target]

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model: Random Forest Regressor ─────────────────────────────────────────────
# n_estimators: number of trees — more trees = more stable but slower.
# max_depth: limits how deep each tree can grow, preventing overfitting.
# min_samples_leaf: a node becomes a leaf once it has ≤ this many samples,
#   which also guards against overfitting on tiny groups.
# n_jobs=-1: use all available CPU cores to train trees in parallel.
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluation ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=== Random Forest Regressor ===")
print(f"MAE  : {mae:.4f}  — average absolute error in delay_rate")
print(f"RMSE : {rmse:.4f}  — penalises large errors more than MAE")
print(f"R²   : {r2:.4f}  — fraction of variance explained (1.0 = perfect)")

# ── Feature importance ─────────────────────────────────────────────────────────
# Random Forest tracks how much each feature reduces impurity across all splits
# in all trees. Higher importance → feature was more useful for predictions.
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10).to_string(index=False))

# ── Plot feature importances (top 15) ─────────────────────────────────────────
top15 = importance_df.head(15)
plt.figure(figsize=(9, 5))
plt.barh(top15["feature"][::-1], top15["importance"][::-1])
plt.xlabel("Importance (mean impurity decrease)")
plt.title("Random Forest — Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=120)
print("\nFeature importance chart saved to rf_feature_importance.png")

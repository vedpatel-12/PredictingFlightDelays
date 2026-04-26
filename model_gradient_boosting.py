"""
Gradient Boosting Model for Predicting Flight Delay Rate
---------------------------------------------------------
Gradient Boosting builds trees sequentially. Each new tree is trained to
correct the residual errors (mistakes) of all previous trees combined. The
model "boosts" performance step by step, nudging predictions in the direction
that reduces loss the most (the gradient of the loss function).

Unlike Random Forest (parallel trees, averaged), Gradient Boosting is additive:
  final prediction = tree_1 + tree_2 + ... + tree_N  (scaled by learning_rate)

Strengths: typically the highest accuracy on tabular data, handles mixed
feature types well, expressive without needing deep trees.
Weakness: more hyperparameters to tune, slower to train than Random Forest
if not carefully configured.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
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

# ── Model: Gradient Boosting Regressor ────────────────────────────────────────
# n_estimators: how many sequential trees to build.
# learning_rate: how much each tree's correction is scaled before being added.
#   Lower rate + more trees = smoother, more accurate, but slower.
# max_depth: depth of each individual tree. Shallow trees (3-5) are typical
#   because GB corrects errors across many trees, not within one deep tree.
# subsample: fraction of training data sampled for each tree (< 1.0 adds
#   randomness, which reduces overfitting — this is "Stochastic GB").
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluation on held-out test set ───────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=== Gradient Boosting Regressor ===")
print(f"MAE  : {mae:.4f}  — average absolute error in delay_rate")
print(f"RMSE : {rmse:.4f}  — penalises large errors more than MAE")
print(f"R²   : {r2:.4f}  — fraction of variance explained (1.0 = perfect)")

# ── Cross-validation ───────────────────────────────────────────────────────────
# 5-fold CV gives a more reliable estimate of generalisation performance than
# a single train/test split, because every sample gets to be in the test fold.
cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"\n5-fold CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# ── Feature importance ─────────────────────────────────────────────────────────
# GB feature importance is the total reduction in loss attributable to each
# feature across all splits and all trees.
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10).to_string(index=False))

# ── Training loss curve ────────────────────────────────────────────────────────
# Shows how training error decreases as more trees are added. A flattening
# curve means additional trees contribute little — useful for tuning n_estimators.
plt.figure(figsize=(8, 4))
plt.plot(model.train_score_, label="Training loss")
plt.xlabel("Number of boosting stages (trees)")
plt.ylabel("Loss (MSE)")
plt.title("Gradient Boosting — Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("gb_training_loss.png", dpi=120)
print("\nTraining loss curve saved to gb_training_loss.png")

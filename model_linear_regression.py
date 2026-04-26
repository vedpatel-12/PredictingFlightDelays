"""
Linear Regression Model for Predicting Flight Delay Rate
---------------------------------------------------------
Linear Regression finds the best-fit straight line through the data by
minimizing the sum of squared errors between predictions and actual values.
It learns a weight (coefficient) for each feature — e.g., how much each
extra unit of avg_airline_delay_rate nudges the predicted delay_rate.

Strengths: fast, interpretable, great baseline.
Weakness: assumes a linear relationship between features and target, which
may not fully capture complex interactions.
"""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("airline_delay_cleaned.csv")

# ── Feature selection (non-leaky columns only) ─────────────────────────────────
# We one-hot encode carrier and airport so the model can learn per-carrier and
# per-airport offsets without imposing any ordinal relationship between them.
features = ["month", "carrier", "airport", "arr_flights",
            "avg_airline_delay_rate", "avg_airport_delay_rate"]
target = "delay_rate"

X = pd.get_dummies(df[features], columns=["carrier", "airport"])
y = df[target]

# ── Train / test split ─────────────────────────────────────────────────────────
# 80 % training, 20 % testing; random_state makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Feature scaling ────────────────────────────────────────────────────────────
# Linear models are sensitive to feature magnitudes. StandardScaler transforms
# each feature to zero mean and unit variance so no single feature dominates.
# We fit only on training data to avoid data leakage into the test set.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Model: Ridge Regression ────────────────────────────────────────────────────
# Ridge is linear regression with L2 regularization (alpha controls strength).
# The penalty shrinks large coefficients, which reduces overfitting when there
# are many one-hot encoded columns (one per carrier / airport).
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# ── Evaluation ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=== Linear Regression (Ridge) ===")
print(f"MAE  : {mae:.4f}  — average absolute error in delay_rate")
print(f"RMSE : {rmse:.4f}  — penalises large errors more than MAE")
print(f"R²   : {r2:.4f}  — fraction of variance explained (1.0 = perfect)")

# ── Top feature coefficients ───────────────────────────────────────────────────
# Larger absolute coefficient → stronger influence on predicted delay_rate.
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)

print("\nTop 10 most influential features:")
print(coef_df.head(10).to_string(index=False))

"""
Linear Regression using scikit-learn (function-based)
Compute final cost, RMSE, MAE, and R²
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# -----------------------------
# 1. LOAD DATA
# -----------------------------
def load_data():
    return pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")


# -----------------------------
# 2. FORM X AND y
# (Remove target leakage)
# -----------------------------
def form_x_and_y(data):
    X = data.drop(["disease_score_fluct", "disease_score"], axis=1).values
    y = data["disease_score_fluct"].values
    return X, y


# -----------------------------
# 3. SCALE FEATURES
# -----------------------------
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
def train_model(X_scaled, y):
    model = LinearRegression()
    model.fit(X_scaled, y)
    return model


# -----------------------------
# 5. PREDICT
# -----------------------------
def predict(model, X_scaled):
    return model.predict(X_scaled)


# -----------------------------
# 6. COMPUTE COST (same as scratch)
# J = (1 / 2m) * Σ (ŷ − y)²
# -----------------------------
def compute_cost(y, y_pred):
    m = len(y)
    return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)


# -----------------------------
# 7. COMPUTE METRICS
# -----------------------------
def compute_metrics(y, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    mae = np.mean(np.abs(y_pred - y))
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2


# -----------------------------
# 8. MAIN FUNCTION
# -----------------------------
def main():
    data = load_data()
    X, y = form_x_and_y(data)

    X_scaled, scaler = scale_features(X)

    model = train_model(X_scaled, y)

    y_pred = predict(model, X_scaled)

    final_cost = compute_cost(y, y_pred)
    rmse, mae, r2 = compute_metrics(y, y_pred)

    print("\nFINAL RESULTS (SCIKIT-LEARN)")
    print("=" * 50)
    print(f"Final Cost (MSE/2)      : {final_cost:.4f}")
    print(f"Root Mean Squared Error : {rmse:.4f}")
    print(f"Mean Absolute Error     : {mae:.4f}")
    print(f"R² Score                : {r2:.4f}")




# -----------------------------
# 9. RUN PROGRAM
# -----------------------------
if __name__ == "__main__":
    main()

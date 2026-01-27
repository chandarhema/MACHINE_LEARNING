"""
Linear Regression from scratch with Gradient Descent
Includes Cost and R² computation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. LOAD DATA
# -----------------------------
def load_data():
    return pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")


# -----------------------------
# 2. FORM X AND y
# -----------------------------
def form_x_and_y(data):
    # Remove target leakage
    X = data.drop(["disease_score_fluct", "disease_score"], axis=1).values
    y = data["disease_score_fluct"].values

    # Scale X
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    # Add bias term
    X = np.c_[np.ones(X.shape[0]), X]

    # Visualization
    plt.scatter(X[:, 1], y)
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Disease_score_fluct")
    plt.title("Feature vs Target")
    plt.grid(True)
    plt.show()

    return X, y


# -----------------------------
# 3. HYPOTHESIS
# -----------------------------
def compute_hypothesis(X, theta):
    return np.dot(X, theta)


# -----------------------------
# 4. COST FUNCTION
# -----------------------------
def compute_cost(X, y, theta):
    m = len(y)
    y_pred = compute_hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


# -----------------------------
# 5. DERIVATIVE or GRADIENT
# -----------------------------
def compute_derivative(X, y, theta):
    m = len(y)
    y_pred = compute_hypothesis(X, theta)
    gradient = (1 / m) * np.dot(X.T, (y_pred - y))
    return gradient


# -----------------------------
# 6. R² FROM SCRATCH
# -----------------------------
def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# -----------------------------
# 7. MAIN (GRADIENT DESCENT)
# -----------------------------
def main():
    data = load_data()
    X, y = form_x_and_y(data)

    # Scale y
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std

    theta = np.zeros(X.shape[1])
    alpha = 0.1
    iterations = 30000

    for i in range(iterations):
        gradient = compute_derivative(X, y_scaled, theta)
        theta = theta - alpha * gradient

        if i % 5000 == 0:
            cost = compute_cost(X, y_scaled, theta)
            print(f"Iteration {i}: Cost = {cost}")
        elif i > 0 and i <= 25:
            cost = compute_cost(X, y_scaled, theta)
            print(f"Iteration {i}: Cost = {cost}")

    # Final prediction (back to original scale)
    y_pred_scaled = compute_hypothesis(X, theta)
    y_pred = y_pred_scaled * y_std + y_mean

    r2 = compute_r2(y, y_pred)

    print("\nFinal theta values:")
    print(theta)
    print(f"\nR² score (from scratch): {r2:.4f}")


# -----------------------------
# 8. RUN
# -----------------------------
if __name__ == "__main__":
    main()

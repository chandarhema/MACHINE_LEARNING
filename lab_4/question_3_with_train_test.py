import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    return data

# --------------------------------------------------
# Form X and Y
# --------------------------------------------------
def form_x_and_y(data):

    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    Y = data["disease_score"].values.reshape(-1, 1)

    # Bias term
    X = np.c_[X, np.ones((X.shape[0], 1))]

    return X, Y

# --------------------------------------------------
# Train-validation split
# --------------------------------------------------
def train_test_split1(X, Y):
    return train_test_split(
        X, Y, train_size=0.8, test_size=0.2, random_state=42
    )

# --------------------------------------------------
# Hypothesis
# --------------------------------------------------
def compute_hypothesis(X, theta):
    return X.dot(theta)

# --------------------------------------------------
# Cost
# --------------------------------------------------
def compute_cost(X, Y, theta):
    m = len(Y)
    y_pred = compute_hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)

# --------------------------------------------------
# Gradient
# --------------------------------------------------
def compute_derivative(X, Y, theta):
    m = len(Y)
    y_pred = compute_hypothesis(X, theta)
    return (1 / m) * np.dot(X.T, (y_pred - Y))

# --------------------------------------------------
# R²
# --------------------------------------------------
def compute_r2(Y, y_pred):
    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    return 1 - (ss_res / ss_tot)

# --------------------------------------------------
# Normal Equation (USING INVERSE ONLY)
# --------------------------------------------------
def theta1(X, Y):
    X_T_X = np.linalg.inv(X.T @ X)
    X_T_Y = X.T @ Y
    theta = X_T_X @ X_T_Y
    return theta

# --------------------------------------------------
# Gradient Descent
# --------------------------------------------------
def gradient_descent(X, Y, alpha=0.0001, num_iters=2000):
    theta = np.zeros((X.shape[1], 1))

    for i in range(num_iters):
        gradient = compute_derivative(X, Y, theta)
        theta = theta - alpha * gradient

        if i <= 25 or i % 200 == 0:
            cost = compute_cost(X, Y, theta)
            print(f"Iteration {i}, Cost: {cost:.4f}")

    return theta

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    data = load_data()
    X, Y = form_x_and_y(data)

    X_train, X_valid, y_train, y_valid = train_test_split1(X, Y)

    # -------- Normal Equation --------
    theta_ne = theta1(X_train, y_train)
    y_pred_ne = compute_hypothesis(X_valid, theta_ne)
    print("R2 (Normal Equation):", compute_r2(y_valid, y_pred_ne))

    # -------- Gradient Descent --------
    theta_gd = gradient_descent(X_train, y_train)
    y_pred_gd = compute_hypothesis(X_valid, theta_gd)
    print("R2 (Gradient Descent):", compute_r2(y_valid, y_pred_gd))

    # -------- sklearn --------
    print("R2 sklearn (NE):", r2_score(y_valid, y_pred_ne))
    print("R2 sklearn (GD):", r2_score(y_valid, y_pred_gd))


if __name__ == "__main__":
    main()

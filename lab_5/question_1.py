"""
Implement Stochastic Gradient Descent algorithm from scratch
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data():
    return pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

# --------------------------------------------------
# Form X and Y
# --------------------------------------------------
def form_x_and_y(data):
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    Y = data["disease_score"].values.reshape(-1, 1)

    # Add bias term
    X = np.c_[X, np.ones((X.shape[0], 1))]
    return X, Y

# --------------------------------------------------
# Train–validation split
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
# Cost function (MSE / 2)
# --------------------------------------------------
def compute_cost(X, Y, theta):
    m = len(Y)
    y_pred = compute_hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)

# --------------------------------------------------
# R2 score (manual)
# --------------------------------------------------
def compute_r2(Y, y_pred):
    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    return 1 - (ss_res / ss_tot)

# --------------------------------------------------
# Stochastic Gradient Descent
# --------------------------------------------------
def stochastic_gradient_descent(X, Y, alpha, num_iters):
    theta = np.zeros((X.shape[1], 1))
    m = X.shape[0]

    cost_history = []
    iteration_history = []

    for i in range(num_iters):
        # Select one random sample
        random_index = np.random.randint(0, m)
        xi = X[random_index].reshape(1, -1)
        yi = Y[random_index]

        # Prediction and error
        y_pred = xi.dot(theta)
        error = y_pred - yi

        # Parameter update (SGD rule)
        for j in range(len(theta)):
            theta[j] = theta[j] - alpha * error * xi[0, j]

    return theta, iteration_history, cost_history

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # Load and prepare data
    data = load_data()
    X, Y = form_x_and_y(data)

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split1(X, Y)

    # Hyperparameters
    alpha = 0.0001
    num_iters = 2000

    # Train using SGD
    theta_sgd, iters, costs = stochastic_gradient_descent(
        X_train, y_train, alpha, num_iters
    )

    # Predictions
    y_train_pred = compute_hypothesis(X_train, theta_sgd)
    y_valid_pred = compute_hypothesis(X_valid, theta_sgd)

    # Evaluation
    print("Training R2:", compute_r2(y_train, y_train_pred))
    print("Validation R2:", compute_r2(y_valid, y_valid_pred))
    print("Validation MSE:", mean_squared_error(y_valid, y_valid_pred))


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    main()

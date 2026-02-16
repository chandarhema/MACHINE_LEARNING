"""
K-fold cross validation.
Implement for K = 10.
Implement from scratch (Gradient Descent),
then use scikit-learn methods.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    print("Dataset shape:", data.shape)
    return data


# --------------------------------------------------
# Shuffle data and split X, Y
# --------------------------------------------------
def shuffle_data(data):
    """
    Shuffle the data.
    data.sample(frac=1) takes 100% of rows and shuffles them.
    reset_index(drop=True) resets indices after shuffling.
    """
    shuffled_data = data.sample(frac=1).reset_index(drop=True)

    X = shuffled_data.drop(columns=["disease_score", "disease_score_fluct"],axis=1).values

    # Add bias column
    X = np.c_[np.ones((X.shape[0], 1)), X]

    Y = shuffled_data["disease_score"].values.reshape(-1, 1)

    return X, Y


# --------------------------------------------------
# Gradient Descent functions (FROM SCRATCH)
# --------------------------------------------------
def gradient_descent(X, Y, alpha=0.00001, iterations=2000):
    """
    Train Linear Regression using Gradient Descent
    """
    m, n = X.shape
    theta = np.zeros((n, 1))   #it creates columns of vectors like [0],[0],[0],[0],[0],[0]

    for _ in range(iterations):
        Y_pred = X @ theta
        gradient = (1 / m) * (X.T @ (Y_pred - Y))
        theta = theta - alpha * gradient

    return theta


def predict(X, theta):
    return X @ theta


def mse(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)


# --------------------------------------------------
# K-Fold Cross Validation (FROM SCRATCH)
# --------------------------------------------------
def k_fold_cross_validation(X, Y):
    k = 10
    total_rows = X.shape[0]
    fold_size = total_rows // k

    mse_list = []

    print("\nK-Fold Cross Validation (From Scratch using Gradient Descent)\n")

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size

        # Test set
        X_test = X[start:end]
        Y_test = Y[start:end]

        # Train set
        X_train = np.vstack((X[:start], X[end:]))
        Y_train = np.vstack((Y[:start], Y[end:]))

        # Train model
        theta = gradient_descent(
            X_train,
            Y_train,
            alpha=0.00001,
            iterations=2000
        )

        # Predict
        Y_pred = predict(X_test, theta)

        # Evaluate
        fold_mse = mse(Y_test, Y_pred)
        mse_list.append(fold_mse)

        print(f"Fold {i+1} MSE: {fold_mse}")

    print("\nAverage MSE (From Scratch):", np.mean(mse_list))


# --------------------------------------------------
# K-Fold Cross Validation (SCIKIT-LEARN)
# --------------------------------------------------
def k_fold_cross_validation_sklearn(X, Y):
    print("\nK-Fold Cross Validation (Using scikit-learn)\n")

    # Remove bias column (sklearn adds intercept internally)
    X_no_bias = X[:, 1:]

    kf = KFold(n_splits=10, shuffle=True)
    mse_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_no_bias), start=1):
        X_train, X_test = X_no_bias[train_idx], X_no_bias[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        model = LinearRegression()
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        fold_mse = mean_squared_error(Y_test, Y_pred)
        mse_list.append(fold_mse)

        print(f"Fold {fold} MSE: {fold_mse}")

    print("\nAverage MSE (scikit-learn):", np.mean(mse_list))


# --------------------------------------------------
# Main function
# --------------------------------------------------
def main():
    data = load_data()
    X, Y = shuffle_data(data)

    k_fold_cross_validation(X, Y)
    k_fold_cross_validation_sklearn(X, Y)


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()

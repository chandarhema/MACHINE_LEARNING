"""Implement a decision regression tree algorithm without using scikit-learn using the diabetes dataset.
Fetch the dataset from scikit-learn library.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


# -------------------------------
# LOAD DATA
# -------------------------------
def load_dataset():
    data = load_diabetes()
    return data.data, data.target


# -------------------------------
# VARIANCE (MSE)
# -------------------------------
def variance(y):
    return np.var(y)


# -------------------------------
# SPLIT ERROR (Weighted variance)
# -------------------------------
def split_error(X_column, y, threshold):

    left_mask = X_column <= threshold
    right_mask = X_column > threshold

    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
        return float('inf')

    n = len(y)
    n_left = len(y[left_mask])
    n_right = len(y[right_mask])

    left_var = variance(y[left_mask])
    right_var = variance(y[right_mask])

    weighted_error = (n_left/n)*left_var + (n_right/n)*right_var

    return weighted_error


# -------------------------------
# BEST SPLIT
# -------------------------------
def best_split(X, y):
    best_error = float('inf')
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        X_column = X[:, feature]
        thresholds = np.unique(X_column)

        for t in thresholds:
            error = split_error(X_column, y, t)

            if error < best_error:
                best_error = error
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold


# -------------------------------
# LEAF VALUE (mean)
# -------------------------------
def leaf_value(y):
    return np.mean(y)


# -------------------------------
# BUILD TREE
# -------------------------------
def build_tree(X, y, depth=0, max_depth=3):

    if len(y) <= 2 or depth >= max_depth:
        return leaf_value(y)

    feature, threshold = best_split(X, y)

    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left_tree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right_tree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)

    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_tree,
        "right": right_tree
    }


# -------------------------------
# PREDICT ONE
# -------------------------------
def predict_one(x, tree):

    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    threshold = tree["threshold"]

    if x[feature] <= threshold:
        return predict_one(x, tree["left"])
    else:
        return predict_one(x, tree["right"])


# -------------------------------
# PREDICT ALL
# -------------------------------
def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])


# -------------------------------
# MSE
# -------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# -------------------------------
# MAIN
# -------------------------------
def main():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TRAIN
    tree = build_tree(X_train, y_train, max_depth=3)

    # TEST
    y_pred = predict(X_test, tree)

    # EVALUATION
    error = mse(y_test, y_pred)

    # TABLE
    result_table = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    print("\nPrediction Table:")
    print(result_table)

    print("\nMSE:", error)

    return {
        "tree": tree,
        "predictions": y_pred,
        "mse": error,
        "table": result_table
    }


if __name__ == "__main__":
    output = main()
"""Implement Adaboost classifier without using scikit-learn. Use the Iris dataset."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# -----------------------------
# Decision stump training
# -----------------------------
def train_stump(X, y, weights):
    n_samples, n_features = X.shape
    best_stump = {}
    min_error = float('inf')

    for feature_i in range(n_features):
        X_column = X[:, feature_i]
        thresholds = np.unique(X_column)

        for threshold in thresholds:
            polarity = 1
            predictions = np.ones(n_samples)
            predictions[X_column < threshold] = -1

            error = np.sum(weights[y != predictions])

            # flip if worse than random
            if error > 0.5:
                error = 1 - error
                polarity = -1

            if error < min_error:
                min_error = error
                best_stump = {
                    "feature_index": feature_i,
                    "threshold": threshold,
                    "polarity": polarity
                }

    return best_stump, min_error


# -----------------------------
# Predict using stump
# -----------------------------
def stump_predict(X, stump):
    n_samples = X.shape[0]
    predictions = np.ones(n_samples)

    feature_values = X[:, stump["feature_index"]]

    if stump["polarity"] == 1:
        predictions[feature_values < stump["threshold"]] = -1
    else:
        predictions[feature_values > stump["threshold"]] = -1

    return predictions


# -----------------------------
# AdaBoost training
# -----------------------------
def adaboost_train(X, y, n_clf):
    n_samples = X.shape[0]
    weights = np.full(n_samples, 1 / n_samples)

    models = []

    for _ in range(n_clf):
        stump, error = train_stump(X, y, weights)

        EPS = 1e-10
        alpha = 0.5 * np.log((1 - error + EPS) / (error + EPS))

        predictions = stump_predict(X, stump)

        # update weights
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)

        stump["alpha"] = alpha
        models.append(stump)

    return models


# -----------------------------
# AdaBoost prediction
# -----------------------------
def adaboost_predict(X, models):
    clf_preds = []

    for model in models:
        preds = stump_predict(X, model)
        clf_preds.append(model["alpha"] * preds)

    y_pred = np.sum(clf_preds, axis=0)
    return np.sign(y_pred)


# -----------------------------
# Load Iris dataset
# -----------------------------
data = load_iris()
X = data.data
y = data.target

# Convert to binary (-1, +1)
y = np.where(y == 0, -1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
models = adaboost_train(X_train, y_train, n_clf=10)

# -----------------------------
# Test model
# -----------------------------
y_pred = adaboost_predict(X_test, models)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)
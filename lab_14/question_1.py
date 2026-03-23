"""Implement Adaboost classifier using scikit-learn. Use the Iris dataset."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# -------------------------------
# LOAD DATASET
# -------------------------------
def load_dataset():
    iris = load_iris()
    data = iris.data
    target = iris.target
    return data, target


# -------------------------------
# EDA (your style)
# -------------------------------
def eda_dataset(data, target):
    # print("Full data:")
    # print(data)

    print("\nFull target:")
    print(target)

    print("\nShape of data:")
    print(data.shape)

    print("\nShape of target:")
    print(target.shape)

    print("\nFirst 5 rows:")
    print(data[:5])

    print("\nLast 5 rows:")
    print(data[-5:])

    print("\nUnique classes:")
    print(np.unique(target))



    return data, target


# -------------------------------
# SPLIT DATA
# -------------------------------
def split_data(data, target):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


# -------------------------------
# MODEL TRAINING (AdaBoost)
# -------------------------------
def train_model(X_train, y_train):

    base_model = DecisionTreeClassifier(max_depth=1)

    model = AdaBoostClassifier(
        estimator=base_model,
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# -------------------------------
# TESTING
# -------------------------------
def test_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    return acc


# -------------------------------
# DISPLAY TABLE
# -------------------------------
def display_results(y_test, y_pred):
    print("\nActual\tPredicted")

    for i in range(len(y_test)):
        correct = y_test[i] == y_pred[i]
        print(f"{y_test[i]}\t{y_pred[i]}\t\t{correct}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    # Load
    data, target = load_dataset()

    # EDA
    data, target = eda_dataset(data, target)

    # Split
    X_train, X_test, y_train, y_test = split_data(data, target)

    # Preprocess
    X_train, X_test = preprocessing(X_train, X_test)

    # Train
    model = train_model(X_train, y_train)

    # Test
    y_pred = test_model(model, X_test)

    # Evaluate
    acc = evaluate_model(y_test, y_pred)

    # Display
    display_results(y_test, y_pred)

    print("\nAccuracy:", acc)

    return {
        "model": model,
        "predictions": y_pred,
        "accuracy": acc
    }


if __name__ == "__main__":
    output = main()
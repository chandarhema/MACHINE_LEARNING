"""Implement logistic regression using scikit-learn for the breast cancer dataset -
 https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data """
"""
Implement Logistic Regression using scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data():
    return pd.read_csv("breast_cancer.csv")

# --------------------------------------------------
# Clean data (drop useless columns)
# --------------------------------------------------
def clean_data(data):
    """
    Removes columns that are not useful for learning.
    - 'id' column is just an identifier (no predictive power)
    - Columns with all missing values are removed
    """
    data = data.drop(columns=["id"], errors="ignore")
    data = data.dropna(axis=1, how="all")
    return data

# --------------------------------------------------
# Form X and y (target dropped from X)
# --------------------------------------------------
def form_x_and_y(data):
    """
    Splits the dataset into:
    X -> input features (all columns except target)
    y -> target labels (diagnosis)

    Diagnosis values:
    M -> 1 (Malignant)
    B -> 0 (Benign)
    """
    X = data.drop(columns=["diagnosis"]).values
    y = data["diagnosis"].map({"M": 1, "B": 0}).values
    return X, y

# --------------------------------------------------
# Handle missing values
# --------------------------------------------------
def handle_missing_values(X):
    """
    Replaces NaN or infinite values with numerical values.
    Logistic Regression cannot work with NaN values.
    """
    return np.nan_to_num(X)

# --------------------------------------------------
# Train–test split
# --------------------------------------------------
def train_test_split1(X, y):
    """
    Splits data into training and testing sets.
    80% -> training
    20% -> testing
    """
    return train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42
    )

# --------------------------------------------------
# Feature scaling
# --------------------------------------------------
def scale_features(X_train, X_test):
    """
    Standardizes features so that:
    - Mean = 0
    - Standard deviation = 1

    Scaling is very important for Logistic Regression
    to ensure faster and stable convergence.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# --------------------------------------------------
# Create logistic regression model
# --------------------------------------------------
def create_model():
    """
    Creates a Logistic Regression classifier.
    max_iter is increased to ensure convergence.
    """
    return LogisticRegression(max_iter=5000)

# --------------------------------------------------
# Train model
# --------------------------------------------------
def train_model(model, X_train, y_train):
    """
    Trains the logistic regression model
    using training data.
    """
    model.fit(X_train, y_train)
    return model

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(model, X_test):
    """
    Predicts class labels (0 or 1)
    for the test dataset.
    """
    return model.predict(X_test)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate_model(y_test, y_pred):
    # Accuracy:
    # Out of all test samples, how many predictions are correct
    # Example: 0.97 means 97% predictions are right
    # Simple overall performance measure
    acc = accuracy_score(y_test, y_pred)

    # Confusion Matrix:
    # Shows where the model is correct and where it is confused
    # [[TN FP]
    #  [FN TP]]
    # TN -> healthy predicted as healthy
    # TP -> cancer predicted as cancer
    # FP -> healthy predicted as cancer
    # FN -> cancer predicted as healthy (dangerous case)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    """
    Main execution flow:
    Load -> Clean -> Prepare -> Split -> Scale -> Train -> Predict -> Evaluate
    """
    data = load_data()
    data = clean_data(data)

    X, y = form_x_and_y(data)
    X = handle_missing_values(X)

    X_train, X_test, y_train, y_test = train_test_split1(X, y)

    X_train, X_test = scale_features(X_train, X_test)

    model = create_model()
    model = train_model(model, X_train, y_train)

    y_pred = predict(model, X_test)

    acc, cm = evaluate_model(y_test, y_pred)

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    main()

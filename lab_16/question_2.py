"""Implement XGBoost classifier and regressor using scikit-learn"""
# =========================
# IMPORTS
# =========================
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score
from xgboost import XGBClassifier, XGBRegressor


# =========================
# 1. CLASSIFICATION (ASD-like binary task)
# =========================
def run_xgboost_classifier():
    print("\n===== XGBOOST CLASSIFIER =====")

    # Example dataset (replace with your gene expression data)
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)


# =========================
# 2. REGRESSION
# =========================
def run_xgboost_regressor():
    print("\n===== XGBOOST REGRESSOR =====")

    # Example dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("RMSE:", rmse)
    print("R2:", r2)


# =========================
# RUN BOTH
# =========================
if __name__ == "__main__":
    run_xgboost_classifier()
    run_xgboost_regressor()
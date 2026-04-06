"""Gradient Boosting vs Lasso (Classification)
Dataset: Weekly (ISLP)
"""

# =========================================================
# IMPORT
# =========================================================
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================================================
# LOAD + SPLIT
# =========================================================
def load_data_weekly():
    data = load_data("Weekly")
    X = data.drop(columns=["Direction"])
    y = data["Direction"].map({"Up": 1, "Down": 0})
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# =========================================================
# MODELS
# =========================================================
def train_gbc(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    return model


def train_lasso_classifier(X_train, y_train):
    model = LogisticRegression(penalty="l1", solver="liblinear")
    model.fit(X_train, y_train)
    return model


# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm


# =========================================================
# COMPARISON FUNCTION (RETURNS EVERYTHING)
# =========================================================
def compare_models(X_train, X_test, y_train, y_test):

    gbc_model = train_gbc(X_train, y_train)
    lasso_model = train_lasso_classifier(X_train, y_train)

    gbc_acc, gbc_cm = evaluate(gbc_model, X_test, y_test)
    lasso_acc, lasso_cm = evaluate(lasso_model, X_test, y_test)

    results = {
        "GradientBoosting": {
            "accuracy": gbc_acc,
            "confusion_matrix": gbc_cm
        },
        "Lasso_Logistic": {
            "accuracy": lasso_acc,
            "confusion_matrix": lasso_cm
        }
    }

    return results, gbc_model, lasso_model


# =========================================================
# MAIN
# =========================================================
def main():
    X, y = load_data_weekly()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = preprocessing(X_train, X_test)

    results, gbc_model, lasso_model = compare_models(
        X_train, X_test, y_train, y_test
    )

    # Print from returned variable
    print("\n--- CLASSIFICATION COMPARISON ---")

    print("\nGradient Boosting:")
    print("Accuracy:", results["GradientBoosting"]["accuracy"])
    print("Confusion Matrix:\n", results["GradientBoosting"]["confusion_matrix"])

    print("\nLasso (L1 Logistic):")
    print("Accuracy:", results["Lasso_Logistic"]["accuracy"])
    print("Confusion Matrix:\n", results["Lasso_Logistic"]["confusion_matrix"])


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()
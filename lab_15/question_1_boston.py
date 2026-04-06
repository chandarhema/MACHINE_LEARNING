"""Gradient Boosting vs Lasso (Regression)
Dataset: Boston (ISLP)
"""

# =========================================================
# IMPORT
# =========================================================
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


# =========================================================
# LOAD + SPLIT
# =========================================================
def load_data_boston():
    data = load_data("Boston")
    X = data.drop(columns=["medv"])
    y = data["medv"]
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# =========================================================
# MODELS
# =========================================================
def train_gbr(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    return model


def train_lasso(X_train, y_train):
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)


# =========================================================
# MAIN
# =========================================================
def main():
    X, y = load_data_boston()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = preprocessing(X_train, X_test)

    gbr = train_gbr(X_train, y_train)
    lasso = train_lasso(X_train, y_train)

    gbr_mse, gbr_r2 = evaluate(gbr, X_test, y_test)
    las_mse, las_r2 = evaluate(lasso, X_test, y_test)

    print("\n--- REGRESSION COMPARISON ---")
    print("\nGradient Boosting:")
    print("MSE:", gbr_mse, "R2:", gbr_r2)

    print("\nLasso:")
    print("MSE:", las_mse, "R2:", las_r2)


if __name__ == "__main__":
    main()
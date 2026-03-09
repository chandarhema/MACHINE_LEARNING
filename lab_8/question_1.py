"""2. Consider the following ER diagram (Cardinality 1 gene encodes multiple proteins):

Create tables for this ER diagram. Insert 5 entries (eg. TP53 gene - encodes for p53alpha,
p53beta, p53gamma proteins ) . (both create, insert queries are needed).

Is a relationship table needed here ? Why or why not ?"""

print(__doc__)
print()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------------------------
# Load and Split Data
# -------------------------------------------------
def load_and_split(filepath):
    df = pd.read_csv(filepath)

    split = int(0.7 * len(df))
    train = df[:split]
    test = df[split:]

    X_train = train.drop(columns=['disease_score','disease_score_fluct'])
    y_train = train['disease_score_fluct']

    X_test = test.drop(columns=['disease_score','disease_score_fluct'])
    y_test = test['disease_score_fluct']

    return X_train, y_train, X_test, y_test


# -------------------------------------------------
# Scale Features
# -------------------------------------------------
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


# -------------------------------------------------
# Add Bias Column
# -------------------------------------------------
def add_bias(X):
    m = X.shape[0]
    return np.c_[np.ones(m), X]


# -------------------------------------------------
# L2 Ridge (Closed Form)
# θ = (XᵀX + λI)^(-1) Xᵀy
# -------------------------------------------------
def ridge_closed_form(X, y, lam):

    y = y.to_numpy().reshape(-1,1)

    n = X.shape[1]
    I = np.eye(n)
    I[0,0] = 0   # do NOT regularize intercept

    theta = np.linalg.inv(
        X.T @ X + lam * I
    ) @ (X.T @ y)

    return theta


# -------------------------------------------------
# L1 Lasso (Gradient Descent - Stable Version)
# -------------------------------------------------
def lasso_gradient_descent(X, y, lam, alpha=0.05, epochs=3000):

    y = y.to_numpy().reshape(-1,1)
    m, n = X.shape

    theta = np.zeros((n,1))

    for _ in range(epochs):

        predictions = X @ theta
        diff = predictions - y

        gradient = (1/m) * (X.T @ diff)

        # L1 regularization term
        reg_term = (lam/m) * np.sign(theta)
        reg_term[0] = 0  # do NOT regularize intercept

        gradient += reg_term

        theta -= alpha * gradient

    return theta


# -------------------------------------------------
# Predict
# -------------------------------------------------
def predict(X, theta):
    return X @ theta


# -------------------------------------------------
# Evaluate
# -------------------------------------------------
def evaluate(y_true, y_pred):
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("R2:", r2_score(y_true, y_pred))


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    X_train, y_train, X_test, y_test = \
        load_and_split("simulated_data_multiple_linear_regression_for_ML.csv")

    # Scale
    X_train, X_test = scale_data(X_train, X_test)

    # Add bias
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    lam = 0.01   # keep small
    alpha = 0.01

    # ---------------- L2 Ridge ----------------
    print("\n===== L2 Ridge =====")
    theta_l2 = ridge_closed_form(X_train, y_train, lam)
    y_pred_l2 = predict(X_test, theta_l2)

    print("Theta L2:", theta_l2.ravel())
    evaluate(y_test, y_pred_l2)

    # ---------------- L1 Lasso ----------------
    print("\n===== L1 Lasso =====")
    theta_l1 = lasso_gradient_descent(X_train, y_train, lam, alpha, epochs=3000)
    y_pred_l1 = predict(X_test, theta_l1)

    print("Theta L1:", theta_l1.ravel())
    evaluate(y_test, y_pred_l1)


if __name__ == "__main__":
    main()
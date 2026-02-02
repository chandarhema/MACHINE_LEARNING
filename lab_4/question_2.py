"""
Linear Regression from Scratch
Dataset: California Housing
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


# -----------------------------
# 1. LOAD DATA
# -----------------------------
def load_data():
    dataset = fetch_california_housing(as_frame=True)
    # print(dataset)
    return dataset.frame


# -----------------------------
# 2. FORM X AND y
# -----------------------------
def form_x_and_y(data):
    X = data.drop("MedHouseVal", axis=1).values
    y = data["MedHouseVal"].values

    # Feature scaling
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    # Add bias term
    X = np.c_[np.ones(X.shape[0]), X]

    return X, y


# -----------------------------
# 3. HYPOTHESIS FUNCTION
# -----------------------------
def hypothesis(X, theta):
    return np.dot(X, theta)


# -----------------------------
# 4. COST FUNCTION (MSE)
# -----------------------------
def compute_cost(X, y, theta):
    m = len(y)
    y_pred = hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)


# -----------------------------
# 5. GRADIENT
# -----------------------------
def compute_gradient(X, y, theta):
    m = len(y)
    y_pred = hypothesis(X, theta)
    return (1 / m) * np.dot(X.T, (y_pred - y))


# -----------------------------
# 6. GRADIENT DESCENT
# -----------------------------
def gradient_descent(X, y, alpha=0.01, iterations=5001):
    theta = np.zeros(X.shape[1])
    cost_history = []

    for i in range(iterations):
        gradient = compute_gradient(X, y, theta)
        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if i % 1000 == 0:
            print(f"Iteration {i} | Cost: {cost:.4f}")

    return theta, cost_history


# -----------------------------
# 7. R² SCORE FROM SCRATCH
# -----------------------------
def r2_score_scratch(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# -----------------------------
# 8. MAIN
# -----------------------------
def main():
    data = load_data()
    X, y = form_x_and_y(data)

    theta, cost_history = gradient_descent(X, y)

    y_pred = hypothesis(X, theta)
    r2 = r2_score_scratch(y, y_pred)

    print("\nFinal Results (From Scratch)")
    print("-----------------------------")
    print("Theta values:")
    print(theta)
    print(f"\nR² score: {r2:.4f}")

    # Cost convergence plot
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Function Convergence")
    plt.grid(True)
    plt.show()


# -----------------------------
# 9. RUN
# -----------------------------
if __name__ == "__main__":
    main()

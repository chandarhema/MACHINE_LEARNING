import numpy as np

def load_data():
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([90.5, 200.4, 300.3, 400.2, 500.1])
    return X, Y


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(X)

    for i in range(num_iters):
        # hypothesis
        y_pred = np.dot(X, theta)

        # gradient
        gradient = (1/m) * np.dot(X.T, (y_pred - y))

        # update theta
        theta = theta - alpha * gradient

        # correct cost
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)

        if i % 100 == 0:
            print(f"Iteration {i} | Cost: {cost:.2f}")

    return theta


def r2_score_scratch(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def main():
    X, Y = load_data()

    # reshape X
    X = X.reshape(-1, 1)

    # initialize theta
    theta = np.array([0.0])

    alpha = 0.1
    num_iters = 1000

    theta_final = gradient_descent(X, Y, theta, alpha, num_iters)

    #  PREDICTIONS
    y_pred = np.dot(X, theta_final)

    #  R²
    r2 = r2_score_scratch(Y, y_pred)

    print("\nFinal theta:", theta_final)
    print("R² score:", r2)


if __name__ == "__main__":
    main()

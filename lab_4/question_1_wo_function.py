import numpy as np

# ---------------- Data ----------------
X = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7]
], dtype=float)
Y = np.array([101, 202, 303, 404], dtype=float)

# reshape X for matrix multiplication
Y = Y.reshape(-1, 1)
print(Y)

# ---------------- Initialization ----------------
theta = np.zeros([4,1])   # parameter
alpha = 0.01               # learning rate
num_iters = 1000     # keep small, otherwise output will be huge
m = X.shape[0]


# print("Initial theta:", theta)
print("-" * 60)

# ---------------- Gradient Descent ----------------
for i in range(num_iters):

    # hypothesis
    y_pred = np.dot(X, theta)

    # gradient computation
    gradient = (1/m) * np.dot(X.T, (y_pred - Y))

    # update theta
    theta = theta - alpha * gradient

    # cost function (Mean Squared Error)
    cost = (1/(2*m)) * np.sum((y_pred - Y)**2)

    # --------- Print EVERYTHING per iteration ---------
    # if i % 200 == 0:
    #     print(f"Iteration {i}")
    #     print(f"y_pred   : {y_pred}")
    #     print(f"gradient : {gradient}")
    #     print(f"theta    : {theta}")
    #     print(f"cost     : {cost}")
    #     print("-" * 60)

# ---------------- Result ----------------
print("Final theta:", theta)
print("Cost:", cost)
print("-" * 60)

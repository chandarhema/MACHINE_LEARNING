# For a design or feature matrix,
# X=[[1,0,2],[0,1,1],[2,1,0],[1,1,1],[0,2,1]]
# Compute the covariance matrix using matrix multiplications.
# Verify your results by using numpy library operations

import numpy as np

X = np.array([[1,0,2],
              [0,1,1],
              [2,1,0],
              [1,1,1],
              [0,2,1]])

# Mean centering
X_mean = X - np.mean(X, axis=0)

# Covariance using matrix multiplication
cov_manual = (X_mean.T @ X_mean) / (X.shape[0] - 1)

print("Covariance (manual):\n", cov_manual)

# Verification using numpy
cov_numpy = np.cov(X, rowvar=False)

print("\nCovariance (numpy):\n", cov_numpy)



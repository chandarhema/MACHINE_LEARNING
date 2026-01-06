# Here is a linear model.y = 2x1 + 3x2 + 3x3 + 4
# The coefficients, represented as theta, is a vector given below
# There are 5 samples represented as a matrix, X, given below
# Compute
import numpy as np
X= np.array([[1,0,2],[0,1,1],[2,1,0],[1,1,1],[0,2,1]])
print(f"X:\n{X}")
theta=np.array([[2],[3],[3]])
print()
print(f"theta:\n{theta}")
print()
# Xtheta=X@theta
Xtheta=np.dot(X,theta)
print(f"Xtheta:\n{Xtheta}")
print()
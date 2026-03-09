import numpy as np
X = np.array([[96], [2], [33], [4]])
y = np.array([0, 0, 1, 1])

indices = np.random.permutation(len(X))
print(indices)

X = X[indices]
y = y[indices]

print(X)
print(y)

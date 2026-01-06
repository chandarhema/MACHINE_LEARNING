#!usr/bin/python
# Implement ATA  -  A = [1 2 3
#                        4 5 6]
import numpy as np
# matrix A
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"A:{A}")

# Transpose of A
A_T = A.T
print()
print(f"A_T:{A_T}")

# Result of A transpose A
print()
print(f"A_T*A:{np.dot(A_T, A)}")
# Define matrix A as a list of lists
A = [
    [1, 2, 3],
    [4, 5, 6]
]

# Transpose of A
A_T = []
for j in range(len(A[0])):        # number of columns in A
    row = []
    for i in range(len(A)):       # number of rows in A
        row.append(A[i][j])
    A_T.append(row)

# Matrix multiplication A_T * A
ATA = []
for i in range(len(A_T)):         # rows of A_T
    row = []
    for j in range(len(A[0])):    # columns of A
        sum_val = 0
        for k in range(len(A)):   # shared dimension
            sum_val += A_T[i][k] * A[k][j]
        row.append(sum_val)
    ATA.append(row)

# Print results
print("Matrix A:")
for r in A:
    print(r)

print("\nTranspose of A (A_T):")
for r in A_T:
    print(r)

print("\nResult of A_T * A:")
for r in ATA:
    print(r)

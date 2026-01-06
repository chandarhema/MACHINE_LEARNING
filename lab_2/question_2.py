# Compute the dot product of two vectors, x and y given below
# x = [2  1  2]T and y = [1  2  2]T .
# What is the meaning of the dot product of two vectors?
# Illustrate that with your own example.
X = [ [2,1,2] ]
Y = [ [1,2,2] ]

X_T = list(map(list, zip(*X)))
Y_T = list(map(list, zip(*Y)))

print("Transpose:", X_T)
print("Transpose:", Y_T)

dot_product = 0

for i in range(len(X[0])):
    # print(len(X[i]))
    dot_product += X[0][i] * Y_T[i][0]
    # print("Dot product:", dot_product)
print("Dot product:", dot_product)

# Mathematical meaning
# The dot product measures how much two vectors point in the same direction.
# Large positive - vectors are aligned
# Zero - vectors are perpendicular
# Negative - vectors are opposite
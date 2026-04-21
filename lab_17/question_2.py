"""Let x1 = [3, 6], x2 = [10, 10].  Use the above “Transform” function to transform
these vectors to a higher dimension and  compute the dot product in a higher dimension.
Print the value.
"""
import numpy as np

def transform_to_high_dimension(x):
    x1,x2 = x
    trans_x = np.array([x1 ** 2, np.sqrt(2) * x1 * x2, x2 ** 2])
    return trans_x

def polynomial_kernel(a, b):
    kernel_value= (a[0]**2)*(b[0]**2) + 2*a[0]*b[0]*a[1]*b[1] + (a[1]**2)*(b[1]**2)
    return kernel_value

x1 = np.array([3, 6])
x2 = np.array([10, 10])

single_point_x1 = transform_to_high_dimension(x1)
single_point_x2 = transform_to_high_dimension(x2)

# ======================================================
# question_1
# ======================================================
print("question_1")
dot_product = np.dot(single_point_x1, single_point_x2)
print("dot product:",dot_product)


# ======================================================
# question_2
# ======================================================
print("question_2")
kernel_value = polynomial_kernel(x1, x2)
print("Kernel value:", kernel_value)

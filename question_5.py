#!/usr/bin/python
# Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
# Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
# What is the value of x1 at which the function value (y) is zero.
# What do you infer from this?
import matplotlib.pyplot as plt
start, stop, num = -10, 10, 100
step = (stop - start) / (num - 1)
x = [start + i * step for i in range(num)]
print(x)
y = [xi * xi for xi in x]
print(y)
# plot
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = x²")
plt.grid(True)
plt.show()
print()

# derivative calculations
points = [-5, -3, 0, 3, 5]
print("x\t\tDerivative")
for z in points:
    print(f"{z}\t\t{2*z}")


"""Compute the derivative of a sigmoid function and visualize it"""
"""Implement sigmoid function in python and visualize it"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

def derivative_sigmoid(x):
    deriv = sigmoid(x) * (1 - sigmoid(x))
    return deriv


def main():
    a = np.random.randint(10, 50)
    # print(a)
    x = np.arange(-a, a)              # NEGATIVE to POSITIVE values
    y = sigmoid(x)
    z = derivative_sigmoid(x)
    # Print values
    for xi, zi in zip(x, z):
        print(f"{xi} -> {zi}")

    # Plot
    plt.plot(x, z)
    plt.xlabel('x')
    plt.ylabel('derivative_Sigmoid(x)')
    plt.title('Sigmoid Function')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

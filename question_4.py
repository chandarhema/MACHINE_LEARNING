#!/usr/bin/python
# Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100]
import math
import matplotlib.pyplot as plt
mu_mean = 0
sigma_sd = 15
start = -100
stop = 100
num = 100
step = (stop - start) / (num - 1)
x = [start + i * step for i in range(num)]
y = [    (1 / (sigma_sd * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((xi - mu_mean) / sigma_sd)**2)
    for xi in x
]

# Plotting
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gaussian PDF (Manual Method)')
plt.grid(True)
plt.show()

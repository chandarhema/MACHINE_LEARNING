#!usr/bin/python
# Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=--10, stop=10, num=100]
import matplotlib.pyplot as plt
start = -10
stop = 10
num = 100
step = (stop - start) / (num - 1)
x = [start + i * step for i in range(num)]
# x_value=np.linspace(-10,10,100)
y=[2*xi*xi + 3*xi + 4 for xi in x]
plt.plot(x,y )
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of x vs. y')
plt.grid(True)
plt.show()
#!usr/bin/python
# Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100]
import matplotlib.pyplot as plt
start = -100
stop = 100
num = 100
step = (stop - start) / (num - 1)
x = [start + i * step for i in range(num)]
# x_value=np.linspace(-100,100,100)
# print(x_value)
y_value=[2*xi + 3 for xi in x]
# print(y_value)
print(f"x_value: {x}, \n y_value: {y_value}")
plt.plot(x,y_value )
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of x vs. y')
plt.grid(True)
plt.show()
"""Let x1 = [3, 6], x2 = [10, 10].  Use the above “Transform” function to transform
these vectors to a higher dimension and  compute the dot product in a higher dimension.
Print the value.
"""
import numpy as np
import matplotlib.pyplot as plt

# Data
blue = np.array([[1,13],[1,18],[2,9],[3,6],[6,3],[9,2],[13,1],[18,1]])
# for x in blue:
#     h1,h2 = x
#     print(h1)
#     print(h2)

red = np.array([[3,15],[6,6],[6,11],[9,5],[10,10],[11,5],[12,6],[16,3]])

# Plot
plt.scatter(blue[:,0], blue[:,1], color='blue', label='Blue')
plt.scatter(red[:,0], red[:,1], color='red', label='Red')

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Original 2D Data")
plt.show()

def Transform(x):
    x1, x2 = x
    trans_x=np.array([x1**2,np.sqrt(2)*x1*x2,x2**2])
    return trans_x

blue_3d = np.array([Transform(x) for x in blue])
red_3d = np.array([Transform(x) for x in red])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(blue_3d[:,0], blue_3d[:,1], blue_3d[:,2], color='blue', label='Blue')
ax.scatter(red_3d[:,0], red_3d[:,1], red_3d[:,2], color='red', label='Red')

ax.set_xlabel("x1^2")
ax.set_ylabel("√2 * x1 * x2")
ax.set_zlabel("x2^2")

plt.title("Transformed 3D Data")
plt.legend()
plt.show()


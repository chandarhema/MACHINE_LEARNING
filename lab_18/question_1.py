"""Consider the following dataset. Implement the RBF kernel.
Check if RBF kernel separates the data well and compare it with the Polynomial Kernel.
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Data
X = np.array([
    [6,5],[6,9],[8,6],[8,8],[8,10],[9,2],[9,5],[10,10],
    [10,13],[11,5],[11,8],[12,6],[12,11],[13,4],[14,8]
])

y = np.array([
    0,0,1,1,1,0,1,1,
    0,1,1,1,0,0,0
])  # Blue=0, Red=1

# Models
rbf_model = SVC(kernel='rbf', gamma=0.5, C=1)
poly_model = SVC(kernel='poly', degree=3, C=1)

# Train
rbf_model.fit(X, y)
poly_model.fit(X, y)


def plot_decision(model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(title)
    plt.show()


plot_decision(rbf_model, "RBF Kernel")
plot_decision(poly_model, "Polynomial Kernel")
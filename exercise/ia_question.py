# A2: Simulated Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# (a) Create vector X with 100 observations from N(0,1)
# ---------------------------------------------------

np.random.seed(1)   # set seed for reproducibility

X = np.random.normal(0, 1, 100)   # mean = 0, std = 1, size = 100

# ---------------------------------------------------
# (b) Create noise vector e from N(0, 0.25)
# variance = 0.25 -> std = sqrt(0.25) = 0.5
# ---------------------------------------------------

e = np.random.normal(0, 0.5, 100)

# ---------------------------------------------------
# (c) Generate y using the model
# y = -1 + 0.5X + e
# ---------------------------------------------------

y = -1 + 0.5 * X + e

# length of y
print("Length of y:", len(y))

# model parameters
# theta_0 = -1 (intercept)
# theta_1 = 0.5 (slope)

# ---------------------------------------------------
# (d) Scatter plot between X and y
# ---------------------------------------------------

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Scatter Plot of X vs y")
plt.show()

# Observation:
# The scatter plot shows a positive linear relationship between X and y.
# The spread around the line is due to the random noise e.

# ---------------------------------------------------
# (e) Fit least squares linear regression model
# ---------------------------------------------------

# reshape X for sklearn
X = X.reshape(-1, 1)

# split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create model
model = LinearRegression()

# train model
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# ---------------------------------------------------
# Plot X_test vs y_test and regression line
# ---------------------------------------------------

plt.scatter(X_test, y_test)

# sort values for smooth line
sorted_index = X_test[:, 0].argsort()

plt.plot(X_test[sorted_index], y_pred[sorted_index], color='red')

plt.xlabel("X_test")
plt.ylabel("y_test")
plt.title("Linear Regression Fit")
plt.show()

# print estimated parameters
print("Estimated intercept:", model.intercept_)
print("Estimated slope:", model.coef_[0])
"""Implement normal equations method from scratch and
compare your results on a simulated dataset (disease score fluctuation as target) and
the admissions dataset
(https://www.kaggle.com/code/erkanhatipoglu/linear-regression-using-the-normal-equation ).
You can compare the results with scikit-learn and your own gradient descent implementation."""

import sys
import numpy as np
import pandas as pd

def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    return data

def form_x_and_y(data):
    X=data.drop(columns=["disease_score","disease_score_fluct"], axis=1).values
    Y=data["disease_score"].values
    X=np.c_[X,np.ones((X.shape[0],1))]
    Y=Y.reshape(-1, 1)
    return X, Y

# def train_test_split():

def compute_hypothesis(X, theta):
    hypothesis = X.dot(theta)
    return hypothesis

def compute_cost(X, Y, theta):
    m = len(Y)
    y_pred = compute_hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)
    return cost

def compute_derivative(X, y, theta):
    m = len(y)
    y_pred = compute_hypothesis(X, theta)
    gradient = (1 / m) * np.dot(X.T, (y_pred - y))
    return gradient

def compute_r2(Y, y_pred):
    ss_res = np.sum((Y - y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    return 1 - (ss_res / ss_tot)

def theta1(X, Y):
    X_T_X = np.linalg.inv(X.T.dot(X))
    X_T_Y = np.dot(X.T,Y)
    theta = np.dot(X_T_X, X_T_Y)
    return theta

# def theta1(X, Y):
#     X_T_X = np.linalg.inv(X.T @ X)
#     X_T_Y = X.T @ Y
#     theta = X_T_X @ X_T_Y
#     return theta

def gradient_descent(X, Y, alpha, num_iters):
    theta = np.zeros((X.shape[1], 1))
    for i in range(num_iters):
        gradient = compute_derivative(X, Y, theta)
        theta = theta - alpha * gradient

        if i % 100 == 0:
            cost = compute_cost(X, Y, theta)
            print(f"Iteration {i}, Cost: {cost}")
    return theta

def main():
    data = load_data()
    X, Y = form_x_and_y(data)
 # --------------------------------------------------
    # for normal equation
 # --------------------------------------------------

    theta=theta1(X,Y)
    print("theta of normal equation \n", theta)

#--------------------------------------------------
    #for gradient descent
#--------------------------------------------------

    alpha = 0.0001
    num_iters = 1000
    gradient = gradient_descent(X, Y, alpha, num_iters)
    print("gradient", gradient)
    print(f"r2:", compute_r2(Y, compute_hypothesis(X, theta)))

if __name__ == "__main__":
    main()



#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    data=pd.read_csv('multiple_linear_regression_dataset.csv')
    return data

def form_x_y(data):
    x=data.drop("Exam_Score",axis=1).values
    y=data["Exam_Score"].values.reshape(-1,1)
    x=np.c_[x,np.ones((len(x),1))]
    # mean = np.mean(x, axis=0)
    # std = np.std(x, axis=0)
    # x = (x - mean) / std
    return x,y

def train_model(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    return x_train,x_test,y_train,y_test

def hypothesis(x,theta):
    y_pred= np.dot(x,theta)
    return y_pred

def cost(theta,x,y):
    m=len(x)
    cost = (1/(2*m))*np.sum((hypothesis(x,theta)-y)**2)
    return cost

def compute_derivative(X, y, theta):
    m = len(y)
    y_pred = hypothesis(X, theta)
    gradient = (1 / m) * np.dot(X.T, (y_pred - y))
    return gradient

def r2_score_scratch(y, y_pred):
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    return 1 - (ss_res / ss_tot)


def main():
    data = load_data()
    x,y = form_x_y(data)
    theta = np.zeros((x.shape[1],1))
    x_train, x_test, y_train, y_test = train_model(x,y)
    # cost_fn=cost(theta,x,y)

    alpha = 0.01
    iters = 5100
    for i in range(iters):
        grad = compute_derivative(x_train, y_train, theta)
        theta = theta - alpha * grad
        cost_fn=cost(theta,x_train,y_train)

        if i%100==0:
            print(f"Iteration {i}: Cost = {cost_fn}")

    y_pred = hypothesis(x_test,theta)
    R2value = r2_score_scratch(y_test, y_pred)  # CHANGED (from scratch)
    print(f"R^2 = {R2value}")


if __name__ == "__main__":
    main()




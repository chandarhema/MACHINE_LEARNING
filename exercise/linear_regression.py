import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    return data

def edata_preprocess(data):
    print("\nHEAD OF THE DATA\n")
    print(data.head())

    print("\nDESCRIPTION OF THE DATA\n")
    print(data.describe())

    print("\nINFO OF THE DATA\n")
    print(data.info())

    print("\nSHAPE OF THE DATA\n")
    print(data.shape)

    print("\nMISSING DATA\n")
    print(data.isnull().sum())

    return data

def form_x_y(data):
    X = data.drop(columns=["disease_score","disease_score_fluct"], axis=1).values
    y = data["disease_score"].values
    return X, y

def training_and_testing(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def model_selection(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("R2 score:", r2)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE score:", mse)


def main():

    data = load_data()

    # print(data)

    data = edata_preprocess(data)

    X,y = form_x_y(data)

    X_train, X_test, y_train, y_test = training_and_testing(X, y)

    X_train, X_test = preprocessing(X_train, X_test)

    model_selection(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

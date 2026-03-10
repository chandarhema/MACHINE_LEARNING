"""Implement a regression decision tree algorithm using scikit-learn for the
simulated dataset.
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data

def eda(data):
    print("\n DATA")
    print(data)

    print("\n HEAD OF THE DATA")
    print(data.head())

    print("\n TAIL OF THE DATA")
    print(data.tail())

    print("\n INFO OF THE DATA")
    print(data.info())

    print("\n DESCRIPTION OF THE DATA ")
    print(data.describe())

    print("\n SHAPE OF THE DATA")
    print(data.shape)

    print("\n MISSING DATA ")
    print(data.isnull().sum())

    print(data.columns)

    return data

def form_x_y(data):

    x = data.iloc[ : , :-2]
    y = data.iloc[ : , -2]
    return x, y

def training_and_testing(x,y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def preprocessing(x_train, x_test):

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

def model_selection(x_train, x_test, y_train, y_test):

    for b in range(2,10):

        print(f"\n TRAINING AND TESTING WITH {b} DECISION TREES")
        model = DecisionTreeRegressor(
            max_depth=b,
            random_state=42
        )

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)

        r2 = r2_score(y_test, y_pred)

        print(f"\n MSE: {mse}")
        print(f"\n R2: {r2}")

    return mse, r2

def main():

    data = load_data()

    data = eda(data)

    x,y = form_x_y(data)

    x_train, x_test, y_train, y_test = training_and_testing(x,y)

    x_train, x_test = preprocessing(x_train, x_test)

    mse,r2 = model_selection(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()


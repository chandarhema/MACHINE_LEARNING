"""Implement Random Forest algorithm for regression and classification using scikit-learn.
Use diabetes and iris datasets."""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def load_data():
    data = pd.read_csv('diabetes_dataset.csv')
    return data

def eda(data):
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
    X = data.drop(columns=['target'])
    y = data['target']
    return X, y

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# def preprocessing(X_train, X_test):
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     return X_train, X_test

def model_selection(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return score, r2

def iris_classification():
     data_iris = pd.read_csv('Iris.csv')

     print("\nHEAD OF THE IRIS DATA\n")
     print(data_iris.head())

     print("\nDESCRIPTION OF THE IRIS DATA\n")
     print(data_iris.describe())

     print("\nINFO OF THE IRIS DATA\n")
     print(data_iris.info())

     print("\nSHAPE OF THE IRIS DATA\n")
     print(data_iris.shape)

     print("\nMISSING DATA\n")
     print(data_iris.isnull().sum())

     X = data_iris.iloc[:, 1:-1]
     y = data_iris.iloc[:, -1]

     print(X)
     print(y)

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # scaler = StandardScaler()
     # X_train = scaler.fit_transform(X_train)
     # X_test = scaler.transform(X_test)

     model = RandomForestClassifier(n_estimators=100, random_state=42)
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)

     acc = accuracy_score(y_test, y_pred)

     print(f'acc: {acc}')

def main():
    data = load_data()

    data = eda(data)

    X,y= form_x_y(data)

    x_train, x_test, y_train, y_test = train_model(X, y)

    # X_train,X_test = preprocessing(x_train, x_test)

    MSE, r2 = model_selection(x_train, x_test, y_train, y_test)

    print("MSE: ", MSE)
    print("R2: ", r2)

    print("="*50)
    iris_classification()

if __name__ == "__main__":
    main()





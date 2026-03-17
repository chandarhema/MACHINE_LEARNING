"""Implement decision tree classifier without using scikit-learn using the iris dataset.
Fetch the iris dataset from scikit-learn library."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():
    Data = pd.read_csv('Iris.csv')
    return Data

def eda_dataset(Data):
    print("whole dataset")
    print(Data)

    print("\ndataset shape")
    print(Data.shape)

    print("\nfeatures of the dataset")
    print(Data.columns)

    print("\nhead of the dataset")
    print(Data.head())

    print("\ntail of the dataset")
    print(Data.tail())

    print("\nDescription of the dataset")
    print(Data.describe())

    print("\nmissing values of the dataset")
    print(Data.isnull().sum())

    return Data

def form_x_y(Data):
    X = Data.iloc[:, :-1]
    y = Data.iloc[:, -1]
    return X, y

def split_data(X, y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test

def preprocessing(X_train,X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test

# def model_selection(X_train,X_test,y_train,y_test):





def main():
    Data = load_dataset()

    data=eda_dataset(Data)

    X,y = form_x_y(data)

    X_train, X_test, y_train, y_test = split_data(X,y)

    X_train, X_test = preprocessing(X_train,X_test)

    print(X_train)
    print("="*100)
    print(X_test)
    print("="*100)

    print()
    print("=*="*100)
    print(y_train)
    print("=*=" * 100)
    print(y_test)





if __name__ == "__main__":
    main()


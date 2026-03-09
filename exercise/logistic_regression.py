import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def load_data():
    data = pd.read_csv("Titanic_train.csv")
    return data


def eda(data):

    print("\nHEAD OF THE DATA\n")
    print(data.head())

    print("\nDESCRIBE THE DATA\n")
    print(data.describe())

    print("\nSHAPE OF THE DATA\n")
    print(data.shape)

    print("\nMISSING DATA\n")
    print(data.isnull().sum())

    # handle missing values correctly (no inplace warning)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    return data


def form_x_y(data):

    # remove unnecessary columns
    X = data.drop(["Survived", "Name", "Cabin", "PassengerId", "Ticket"], axis=1)

    y = data["Survived"]

    return X, y


def train_model(X, y):

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def feature_encoding(X_train, X_test):

    # encode Sex column
    X_train["Sex"] = X_train["Sex"].map({"male": 1, "female": 0})
    X_test["Sex"] = X_test["Sex"].map({"male": 1, "female": 0})

    # one-hot encoding for Embarked
    X_train = pd.get_dummies(X_train, columns=["Embarked"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["Embarked"], drop_first=True)

    # align columns to prevent mismatch
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test


def preprocessing(X_train, X_test):

    scaler = StandardScaler()

    # fit only on training data
    X_train = scaler.fit_transform(X_train)

    # transform test data
    X_test = scaler.transform(X_test)

    return X_train, X_test


def model_selection(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=10000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    print("ACCURACY SCORE:", acc_score)

    dt_model = DecisionTreeClassifier(random_state=42)

    dt_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)

    dt_acc = accuracy_score(y_test, dt_pred)

    print("Decision Tree Accuracy:", dt_acc)

    return model

def model_selection1(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=10000)

    scores = cross_val_score(model, X_train, y_train, cv=10)

    print("\n10-FOLD CROSS VALIDATION SCORES:\n", scores)

    print("\nMEAN CV ACCURACY:", scores.mean())

    print("STD:", scores.std())


def main():

    data = load_data()

    data = eda(data)

    X, y = form_x_y(data)

    X_train, X_test, y_train, y_test = train_model(X, y)

    X_train, X_test = feature_encoding(X_train, X_test)

    X_train, X_test = preprocessing(X_train, X_test)

    model = model_selection(X_train, X_test, y_train, y_test)

    model1 = model_selection1(X_train, X_test, y_train, y_test)




if __name__ == "__main__":
    main()
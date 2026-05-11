import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# -------------------------------
# Load data
# -------------------------------
def load_dataset():
    data = pd.read_csv("OJ.csv")
    return data


# -------------------------------
# EDA (same style as yours)
# -------------------------------
def eda_preprocess(data):
    print("\nHEAD OF DATA\n")
    print(data.head())

    print("\nDESCRIPTION\n")
    print(data.describe())

    print("\nINFO\n")
    print(data.info())

    print("\nSHAPE\n")
    print(data.shape)

    print("\nMISSING VALUES\n")
    print(data.isnull().sum())

    return data


# -------------------------------
# Form X and y
# -------------------------------
def form_X_y(data):
    # Convert target
    data['Purchase'] = data['Purchase'].map({'CH': 1, 'MM': 0})

    X = data.drop(columns=['Purchase'])
    y = data['Purchase']

    # Handle categorical
    X = pd.get_dummies(X, drop_first=True)

    return X.values, y.values


# -------------------------------
# Train-test split (800 train)
# -------------------------------
def train_test_data(X, y):
    return train_test_split(X, y, train_size=800, random_state=42)


# -------------------------------
# Scaling
# -------------------------------
def preprocessing(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


# -------------------------------
# Linear SVM
# -------------------------------
def linear_svm_model(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear', C=0.01)

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("\n--- Linear SVM (C=0.01) ---")
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))


# -------------------------------
# RBF SVM
# -------------------------------
def rbf_svm_model(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf')  # default gamma

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("\n--- RBF SVM ---")
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))


# -------------------------------
# Main pipeline
# -------------------------------
def main():

    data = load_dataset()

    data = eda_preprocess(data)

    X, y = form_X_y(data)

    X_train, X_test, y_train, y_test = train_test_data(X, y)

    X_train, X_test = preprocessing(X_train, X_test)

    linear_svm_model(X_train, X_test, y_train, y_test)

    rbf_svm_model(X_train, X_test, y_train, y_test)


# Run
if __name__ == "__main__":
    main()
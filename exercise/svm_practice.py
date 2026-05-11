import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


def load_data():
    df = pd.read_csv("OJ.csv")
    return df


def edata(df):
    print("DESCRIPTION OF THE DATA")
    print(df.describe())

    print("\nHead of the data")
    print(df.head())

    print("\nColumn names")
    print(df.columns)

    print("\nData types")
    print(df.dtypes)

    print("\nMissing values")
    print(df.isnull().sum())

    return df


def target_split(data):
    X = data.drop(columns=["Purchase"])
    y = data.Purchase.map({'CH': 0, 'MM': 1})
    print("\ny.value_counts():", y.value_counts())
    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def training_and_split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=800, random_state=42
    )
    return X_train, X_test, y_train, y_test


def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def model_selection(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear', C=0.01)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    return model, train_acc, test_acc


def model_selection2(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    return model, train_acc, test_acc


# 🔥 CLEAN SVM PLOT
def plot_svm(X, y, model, title):

    # Reduce to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Train model on 2D data
    model.fit(X_2d, y)

    # Mesh grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 🎨 Plot
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)

    plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y,
        cmap='coolwarm',
        edgecolors='black',
        s=40
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def main():
    df = load_data()

    data = edata(df)

    X, y = target_split(data)

    x_train, x_test, y_train, y_test = training_and_split_data(X, y)

    X_train, X_test = preprocessing(x_train, x_test)

    # 🔹 Linear SVM
    model1, train1, test1 = model_selection(X_train, X_test, y_train, y_test)

    print("\n--- Linear SVM (C=0.01) ---")
    print("Train Accuracy:", train1)
    print("Test Accuracy:", test1)

    # 🔹 RBF SVM
    model2, train2, test2 = model_selection2(X_train, X_test, y_train, y_test)

    print("\n--- RBF SVM (default gamma) ---")
    print("Train Accuracy:", train2)
    print("Test Accuracy:", test2)

    # 🔥 Plot clean graphs
    plot_svm(X_train, y_train, SVC(kernel='linear', C=0.01),
             "Linear SVM Decision Boundary")

    plot_svm(X_train, y_train, SVC(kernel='rbf'),
             "RBF SVM Decision Boundary")


if __name__ == "__main__":
    main()
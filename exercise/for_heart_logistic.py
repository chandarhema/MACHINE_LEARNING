import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data():
    data = pd.read_csv('Heart.csv')
    return data

def eda(data):

    print("\n DATA \n")
    print(data)

    print("\n HEAD OF THE DATA \n")
    print(data.head())

    print("\n TAIL OF THE DATA \n")
    print(data.tail())

    print("\n INFO OF THE DATA \n")
    print(data.info())

    print("\n DESCRIPTION OF THE DATA \n ")
    print(data.describe())

    print("\n SHAPE OF THE DATA \n")
    print(data.shape)

    print("\n MISSING DATA \n ")
    print(data.isnull().sum())

    data["Ca"] = data["Ca"].fillna(data["Ca"].median())
    data["Thal"] = data["Thal"].fillna(data["Thal"].mode()[0])

    return data

def form_x_y(data):
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def feature_encoding(X_train, X_test):

    categorical_features = ["Thal","ChestPain"]
    encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)

    X_train_cat = encoder.fit_transform(X_train[categorical_features])
    X_test_cat = encoder.transform(X_test[categorical_features])

    X_train_numeric = X_train.drop(columns=categorical_features).values
    X_test_numeric = X_test.drop(columns=categorical_features).values

    X_train_final = np.hstack((X_train_numeric, X_train_cat))
    X_test_final = np.hstack((X_test_numeric, X_test_cat))

    return X_train_final, X_test_final

def preprocessing(X_train_final, X_test_final):

    scaler = StandardScaler()

    X_train_final = scaler.fit_transform(X_train_final)

    X_test_final = scaler.transform(X_test_final)

    return X_train_final, X_test_final

def model_selection(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("=-=" * 30)
    print("ACCURACY OF THE MODEL")
    print(accuracy)
    print("=-=" * 30)

    model1 = DecisionTreeClassifier(random_state=42,splitter='random',max_depth=5)

    model1.fit(X_train, y_train)

    y_pred = model1.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("=+"*50)
    print("ACCURACY OF THE MODEL OF DECISION TREE")
    print(accuracy)
    print("=+" * 50)

    return accuracy

def main():

    data = load_data()

    data = eda(data)

    X, y = form_x_y(data)

    X_train, X_test, y_train, y_test = train_model(X, y)

    X_train_final, X_test_final = feature_encoding(X_train, X_test)

    X_train_final, X_test_final = preprocessing(X_train_final, X_test_final)

    accuracy = model_selection(X_train_final, X_test_final, y_train, y_test)

if __name__ == "__main__":
    main()
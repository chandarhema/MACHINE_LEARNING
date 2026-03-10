"""Build a classification model for wisconsin dataset using Ridge and Lasso classifier using scikit-learn"""

import pandas as pd

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data():
    data = pd.read_csv("data.csv")
    return data

def edata(data):

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

def form_x_and_y(data):
    x= data.drop(columns=["Unnamed: 32","diagnosis","id"],axis=1)
    # print(x.columns)
    # print(x.shape)
    y= data["diagnosis"]
    le = LabelEncoder()
    y=le.fit_transform(y)
    return x,y

def training_and_testing(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def preprocessing(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

def model_selection(x_train, x_test, y_train, y_test):

    # Ridge Model
    ridge = Ridge()
    ridge.fit(x_train, y_train)

    ridge_pred = ridge.predict(x_test)

    # Convert regression output to class labels
    ridge_pred = [1 if i > 0.5 else 0 for i in ridge_pred]

    ridge_acc = accuracy_score(y_test, ridge_pred)

    print("\nRidge Accuracy:", ridge_acc)


    # Lasso Model
    lasso = LogisticRegression(penalty='l1', solver='liblinear')

    lasso.fit(x_train, y_train)

    lasso_pred = lasso.predict(x_test)

    lasso_pred = [1 if i > 0.5 else 0 for i in lasso_pred]

    lasso_acc = accuracy_score(y_test, lasso_pred)

    print("Lasso Accuracy:", lasso_acc)

def main():

    data = load_data()

    data = edata(data)

    x,y = form_x_and_y(data)

    x_train, x_test, y_train, y_test = training_and_testing(x,y)

    x_train, x_test = preprocessing(x_train, x_test)

    model_selection(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()







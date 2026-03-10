"""Implement a classification decision tree algorithm using scikit-learn for the sonar  dataset.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_data():
    # column_names = [f"feature_{i}" for i in range(61)]
    data = pd.read_csv('Copy of sonar data.csv')
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

    # print(data.columns)

    return data

def form_x_y(data):

    x = data.iloc[ : , :-1]
    y = data.iloc[ : , -1]
    le = LabelEncoder()
    y=le.fit_transform(y)
    # print(y)
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

        print(f"\n TRAINING AND TESTING WITH {b} DECISION TREES CLASSIFIER")

        model = DecisionTreeClassifier(
            max_depth=b,
            random_state=42
        )

        model.fit(x_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(x_test))

        print(f'Accuracy score: {accuracy}')

    return accuracy

def main():

    data = load_data()

    data = eda(data)

    x,y = form_x_y(data)

    x_train, x_test, y_train, y_test = training_and_testing(x,y)

    x_train, x_test = preprocessing(x_train, x_test)

    accuracy = model_selection(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()


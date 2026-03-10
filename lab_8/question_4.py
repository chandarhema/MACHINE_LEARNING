"""Use breast_cancer.csv (https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv)
and use scikit learn methods, OrdinalEncoder, OneHotEncoder(sparse=False), LabelEncoder to implement
 complete Logistic Regression Model.
Good reference: https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
    column_names = [f"feature_{i}" for i in range(10)]
    data = pd.read_csv(url,header=None,names=column_names)
    return data

def eda(data):

    print("\n DATA")
    print(data)

    print("\n HEAD OF THE DATA")
    print(data.head(286))

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

    x = data.drop(columns=['feature_9'], axis=1)
    y = data['feature_9']
    le = LabelEncoder()
    y = le.fit_transform(y)
    return x,y

def training_and_testing(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def feature_encoding(x_train, x_test):

    #filling the empty places with most repeated values

    x_train = x_train.fillna(x_train.mode().iloc[0])
    x_test = x_test.fillna(x_train.mode().iloc[0])

    encoder = OneHotEncoder(sparse_output=False)

    encoder.fit(x_train)

    encoded_train = encoder.transform(x_train)
    encoded_test = encoder.transform(x_test)

    # print(encoded_train)
    # print(encoded_test)                     #after changing the value how it will be that's what we will see here
    return encoded_train, encoded_test

def preprocessing(x_train, x_test):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_test = scaler.transform(x_test)
    return scaled_train, scaled_test

def model_selection(x_train, x_test, y_train, y_test):
    model = LogisticRegression(max_iter=10000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return model

def main():

    data = load_data()

    data = eda(data)

    x,y = form_x_y(data)

    x_train, x_test, y_train, y_test = training_and_testing(x,y)

    encoded_train, encoded_test = feature_encoding(x_train, x_test)

    scaled_train, scaled_test = preprocessing(encoded_train, encoded_test)

    model = model_selection(scaled_train, scaled_test, y_train, y_test)


if __name__ == "__main__":
    main()




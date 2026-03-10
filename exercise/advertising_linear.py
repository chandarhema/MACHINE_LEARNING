import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data():

    data = pd.read_csv("advertising.csv")
    return data

def eda_analysis(data):

    print("\n DATA \n")
    print(data)

    print("\n HEAD OF THE DATA \n")
    print(data.head())

    print ("\n TAIL OF THE DATA \n")
    print(data.tail())

    print ("\n INFO OF THE DATA \n")
    print(data.info())

    print("\n DESCRIPTION OF THE DATA \n ")
    print(data.describe())

    print("\n SHAPE OF THE DATA \n")
    print(data.shape)

    print ("\n MISSING DATA \n ")
    print(data.isnull().sum())

    return data

def form_x_and_y(data):
    X=data.drop(columns=["Sales"],axis=1).values
    y=data.Sales.values
    return X,y

def training_and_testing(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def model_selection(X_train, X_test, y_train, y_test):
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n RMSE = ", rmse)
    print("\n R2 = ", r2)
    return model

def main():

    data = load_data()

    data = eda_analysis(data)

    X,y= form_x_and_y(data)

    X_train, X_test, y_train, y_test = training_and_testing(X,y)

    X_train, X_test= preprocessing(X_train, X_test)

    model_selection(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()




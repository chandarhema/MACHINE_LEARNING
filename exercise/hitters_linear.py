import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error , r2_score

def load_data():
    data = pd.read_csv('Hitters.csv')
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

    data = data.dropna(subset=["Salary"])
    # print(data.isnull().sum())
    print(data.shape)
    return data

def form_x_y(data):
    X = data.drop(["Salary"], axis=1)
    y = data["Salary"]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def feature_encoding(X_train, X_test):

    categorical_features = ["League","Division","NewLeague"]

    encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)

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
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("LINEAR REGRESSION MODEL")
    print("=" * 50 )
    print("\n MSE = ", mse)
    print("\n R2 = ", r2)
    print("=" * 50)

    model1=DecisionTreeRegressor(random_state=42,max_depth=4)
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("DECISION TREE MODEL")
    print("=" * 50 )
    print("\n MSE = ", mse)
    print("\n R2 = ", r2)
    print("=" * 50)

    model2=RandomForestRegressor(n_estimators=100, random_state=42)
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("=" * 50 )
    print("RANDOM FOREST REGRESSION MODEL")
    print("\n MSE = ", mse)
    print("\n R2 = ", r2)
    print("=" * 50)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.show()

    residuals = y_test - y_pred

    plt.scatter(y_pred, residuals)
    plt.axhline(y=0)
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

    # models = ["Linear Regression", "Decision Tree", "Random Forest"]
    # r2_scores = [r2_lr, r2_dt, r2_rf]
    #
    # plt.bar(models, r2_scores)
    # plt.ylabel("R2 Score")
    # plt.title("Model Performance Comparison")
    # plt.show()

def main():

    # load dataset
    data = load_data()

    # exploratory data analysis
    data = eda(data)

    # create features and target
    X, y = form_x_y(data)

    # split dataset
    X_train, X_test, y_train, y_test = train_model(X, y)

    # encode categorical features
    X_train_final, X_test_final = feature_encoding(X_train, X_test)

    # scale features
    X_train_final, X_test_final = preprocessing(X_train_final, X_test_final)

    # train and evaluate models
    model_selection(X_train_final, X_test_final, y_train, y_test)


if __name__ == "__main__":
    main()
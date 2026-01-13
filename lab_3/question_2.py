"""Implement a linear regression model using scikit-learn
for the simulated dataset - simulated_data_multiple_linear_regression_for_ML.csv  -
to predict the “disease_score_fluct” from multiple clinical parameters.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

# 1. Load data
def load_data():
    X=data.drop("disease_score_fluct", axis=1)
    y=data["disease_score_fluct"]
    return X,y,data,data.head()

# 2. Split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3)

# 3. Impute missing values
def impute_data(X_train, X_test):
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test

# 4. Standardize data
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# 5. Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 6. Test model
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return y_pred, r2_score(y_test, y_pred)

# Main
def main():
    X, y, frame, description = load_data()

    print("\nDataset Description:\n")
    print(description)

    print("\nFull Dataset (first 10 rows):\n")
    print(frame.head(10))

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    X_train, X_test = impute_data(X_train, X_test)
    # print("\nAfter Imputation (first 5 rows of X_train):\n", X_train)

    X_train, X_test = standardize_data(X_train, X_test)
    # print("\nAfter Standardization (first 5 rows of X_train):\n", X_train)

    model = train_model(X_train, y_train)

    # print("\nLearned Coefficients:\n", model.coef_)
    # print("Intercept:", model.intercept_)

    y_pred, r2 = test_model(model, X_test, y_test)

    print("\nPredicted values (or) disease_score” :\n", y_pred)
    print("\nActual values:\n", y_test.values)

    print("\nPredicted vs Actual values:\n")
    for pred, actual in zip(y_pred, y_test.values):
        print(f"Predicted: {pred:}   \t  Actual: {actual:}")

    print("\nR2 Score:", r2)

if __name__ == "__main__":
    main()




import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data


def eda(data):
    print("\nDATA")
    print(data)

    print("\nHEAD")
    print(data.head())

    print("\nTAIL")
    print(data.tail())

    print("\nINFO")
    print(data.info())

    print("\nDESCRIPTION")
    print(data.describe())

    print("\nSHAPE")
    print(data.shape)

    print("\nMISSING VALUES")
    print(data.isnull().sum())

    print("\nCOLUMNS")
    print(data.columns)

    return data


def form_x_y(data):
    x = data.iloc[:, :-2]
    y = data.iloc[:, -2]
    return x, y


def training_and_testing(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


def preprocessing(x_train, x_test):
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Convert back to DataFrame for SHAP (important)
    x_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    return x_train, x_test


def model_selection(x_train, x_test, y_train, y_test):

    best_model = None
    best_r2 = -1

    for depth in range(2, 10):

        print(f"\nTRAINING WITH DEPTH = {depth}")

        model = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {mse}")
        print(f"R2: {r2}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    print("\nBest model selected")

    return best_model


def apply_shap(model, x_test):
    print("\nApplying SHAP...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    print("Generating SHAP beeswarm plot...")

    shap.summary_plot(shap_values, x_test)


def main():
    data = load_data()

    data = eda(data)

    x, y = form_x_y(data)

    x_train, x_test, y_train, y_test = training_and_testing(x, y)

    x_train, x_test = preprocessing(x_train, x_test)

    best_model = model_selection(x_train, x_test, y_train, y_test)

    apply_shap(best_model, x_test)


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sympy.abc import theta


def question1():
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split

        data = pd.read_csv("advertising.csv")

        print("HEAD OF THE DATA")
        print(data.head())

        print("DESCRIBE THE DATA")
        print(data.describe())

        print("SHAPE OF THE DATA")
        print(data.shape)

        print("MISSING VALUES")
        print(data.isnull().sum())

        print("COLUMN NAMES")
        print(data.columns)

        # Features and target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------- TRAINING (Normal Equation) --------
        theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        # Prediction
        y_pred = np.dot(X_test, theta)

        # Cost Function
        m = len(y_test)
        cost = (1 / (2 * m)) * np.sum((y_pred - y_test) ** 2)

        # Gradient
        gradient = (1 / (2 * m)) * np.dot(X_test.T, (y_pred - y_test))

        # R2 Score
        y_mean = np.mean(y_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_mean) ** 2)

        r2 = 1 - (ss_res / ss_tot)

        print("\nRESULTS")
        print("=" * 50)
        print("Theta:", theta)
        print("Cost Function:", cost)
        print("Gradient:", gradient)
        print("R2 Score:", r2)
        print("=" * 50)

def main():
    question1()


if __name__ == "__main__":
    main()
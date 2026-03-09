"""Implement bagging regressor and classifier using scikit-learn. Use diabetes and iris datasets.
"""


from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


# -------------------------------------------------------
# BAGGING REGRESSOR (Diabetes)
# -------------------------------------------------------
def bagging_regressor_diabetes():

    # Load dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Base estimator
    base_model = DecisionTreeRegressor()

    # Bagging Regressor
    bag_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=100,
        random_state=42
    )

    # Train
    bag_model.fit(X_train, y_train)

    # Predict
    y_pred = bag_model.predict(X_test)

    # Evaluate
    print("----- Bagging Regressor (Diabetes) -----")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print()


# -------------------------------------------------------
# BAGGING CLASSIFIER (Iris)
# -------------------------------------------------------
def bagging_classifier_iris():

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Base estimator
    base_model = DecisionTreeClassifier()

    # Bagging Classifier
    bag_model = BaggingClassifier(
        estimator=base_model,
        n_estimators=100,
        random_state=42
    )

    # Train
    bag_model.fit(X_train, y_train)

    # Predict
    y_pred = bag_model.predict(X_test)

    # Evaluate
    print("----- Bagging Classifier (Iris) -----")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print()


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    bagging_regressor_diabetes()
    bagging_classifier_iris()


if __name__ == "__main__":
    main()
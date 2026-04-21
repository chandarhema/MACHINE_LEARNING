import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


# -------------------------------
# LOAD + EDA + CLEAN
# -------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)

    print("\n--- BASIC INFO ---")
    print(df.head())
    print(df.info())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    # Replace 'NA' with NaN
    df = df.replace("NA", np.nan)

    print("\n--- AFTER REPLACING NA ---")
    print(df.isnull().sum())

    # Fill missing values
    df = df.fillna(df.mode().iloc[0])

    # Convert target
    df["AHD"] = df["AHD"].map({"Yes": 1, "No": 0})

    # Encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    print("\n--- AFTER ENCODING ---")
    print(df.head())

    X = df.drop("AHD", axis=1)
    y = df["AHD"]

    return X, y


# -------------------------------
# SPLIT
# -------------------------------
def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


# -------------------------------
# TRAIN
# -------------------------------
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# -------------------------------
# METRICS
# -------------------------------
def metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP) if (TP+FP)!=0 else 0
    sens = TP / (TP + FN) if (TP+FN)!=0 else 0
    spec = TN / (TN + FP) if (TN+FP)!=0 else 0
    f1 = (2 * prec * sens / (prec + sens)) if (prec+sens)!=0 else 0

    return TP, TN, FP, FN, acc, prec, sens, spec, f1


# -------------------------------
# THRESHOLD TEST
# -------------------------------
def test_thresholds(y_test, y_prob):
    for t in [0.3, 0.5, 0.7]:
        y_pred = (y_prob >= t).astype(int)

        TP, TN, FP, FN, acc, prec, sens, spec, f1 = metrics(y_test.values, y_pred)

        print("\nThreshold:", t)
        print("TP:", TP, "FP:", FP)
        print("FN:", FN, "TN:", TN)

        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Sensitivity:", sens)
        print("Specificity:", spec)
        print("F1:", f1)


# -------------------------------
# ROC
# -------------------------------
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print("\nAUC:", roc_auc)

    plt.plot(fpr, tpr, label="AUC = " + str(round(roc_auc, 2)))
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# -------------------------------
# MAIN
# -------------------------------
def main():
    X, y = load_data("Heart.csv")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:,1]

    test_thresholds(y_test, y_prob)

    plot_roc_curve(y_test, y_prob)


# -------------------------------
# RUN
# -------------------------------
main()
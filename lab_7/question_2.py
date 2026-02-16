"""Compute SONAR classification results with and without data pre-processing (data normalization).
Perform data pre-processing with your implementation and
with scikit-learn methods and compare the results."""


print(__doc__)

"""AS I DID THE FIRST QUESTION ALSO LIKE DATA PREPROCESSING AND POST PROCESSING
I DID THIS QUESTION WITH DIFFERENT DATASET"""

print()

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ---------------- LOAD ----------------
def load_data():
    return pd.read_csv("IRIS.csv")

# ---------------- SPLIT X,Y ----------------
def form_x_y(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# ---------------- LABEL ENCODING ----------------
def encode_labels(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y)

# ---------------- STANDARDIZATION ----------------
def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# ---------------- NORMALIZATION ----------------
def normalize_data(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

# ---------------- CROSS VALIDATION ----------------
def run_cv(X, y):
    model = LogisticRegression(max_iter=1000)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold)
    return scores

# ---------------- MAIN ----------------
def main():
    data = load_data()
    X, y = form_x_y(data)

    # Encode labels
    y = encode_labels(y)

    # ---- RAW DATA ----
    raw_scores = run_cv(X, y)
    print("="*50)
    print("--- No Preprocessing ---")
    print("="*50)

    print("Fold accuracy:", raw_scores)
    print("Mean accuracy:", raw_scores.mean())

    # ---- STANDARDIZED ----
    X_std = standardize_data(X)
    std_scores = run_cv(X_std, y)
    print()
    print("="*50)
    print("--- Standardization ---")
    print("="*50)

    print("Fold accuracy:", std_scores)
    print("Mean accuracy:", std_scores.mean())

    # ---- NORMALIZED ----
    X_norm = normalize_data(X)
    norm_scores = run_cv(X_norm, y)
    print()
    print("="*50)
    print("--- Normalization ---")
    print("="*50)

    print("Fold accuracy:", norm_scores)
    print("Mean accuracy:", norm_scores.mean())

if __name__ == "__main__":
    main()


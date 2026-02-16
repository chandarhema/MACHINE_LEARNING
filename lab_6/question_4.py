"""Data standardization - scale the values such that mean of new dist = 0 and sd = 1.
Implement code from scratch."""
print(__doc__)

import pandas as pd

def load_data():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data

def form_x(data):
    x = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    return x.astype(float)

def normalize_minmax(x):
    rows, cols = x.shape
    x_norm = x.copy()

    print("="*50)
    print("--- MIN-MAX NORMALIZATION ---")
    print("="*50)
    print("Original X:\n", x)

    for col in range(cols):
        mn = x[0][col]
        mx = x[0][col]

        # find min and max
        for row in range(rows):
            if x[row][col] < mn:
                mn = x[row][col]
            if x[row][col] > mx:
                mx = x[row][col]

        denom = mx - mn
        if denom == 0:
            denom = 1

        # normalize
        for row in range(rows):
            x_norm[row][col] = (x[row][col] - mn) / denom

    print("\nNormalized X:\n", x_norm)
    return x_norm

def standardize_data(x):
    rows, cols = x.shape
    x_std = x.copy()

    print("\n--- STANDARDIZATION (Z-SCORE) ---")
    print("Original X:\n", x)

    for col in range(cols):

        # compute mean
        total = 0
        for row in range(rows):
            total += x[row][col]
        mean = total / rows

        # compute std
        var_sum = 0
        for row in range(rows):
            var_sum += (x[row][col] - mean) ** 2
        variance = var_sum / rows
        std = variance ** 0.5

        if std == 0:
            std = 1

        # standardize
        for row in range(rows):
            x_std[row][col] = (x[row][col] - mean) / std

        print(f"\nColumn {col} -> Mean: {mean:.3f}, Std: {std:.3f}")

    print("\nStandardized X:\n", x_std)
    return x_std

def main():
    data = load_data()
    x = form_x(data)

    x_norm = normalize_minmax(x)
    x_std = standardize_data(x)

if __name__ == "__main__":
    main()

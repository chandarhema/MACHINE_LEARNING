"""Data normalization - scale the values between 0 and 1. Implement code from scratch"""

# import numpy as np
# x=(50,56,598,59,600,60,51,51,502)
# for i in x:
#     # a=min(x)
#     # b=max(x)
#     # print(a,b)
#     xnew= (i - min(x)) / (max(x) - min(x))
#     print(xnew)

import pandas as pd
import numpy as np
def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    return data

def form_x_y(data):
    x=data.drop(columns=["disease_score","disease_score_fluct"]).values
    y=data["disease_score"].values
    return x,y

def normalize(x):
    rows = x.shape[0]
    cols = x.shape[1]

    print("Original X:\n", x.copy())

    for i in range(cols):   # column loop

        # find min
        min_value = x[0][i]
        for j in range(rows):
            if x[j][i] < min_value:
                min_value = x[j][i]

        # find max
        max_value = x[0][i]
        for j in range(rows):
            if x[j][i] > max_value:
                max_value = x[j][i]

        print("\nColumn", i, "Min:", min_value, "Max:", max_value)

        denominator = max_value - min_value
        if denominator == 0:
            denominator = 1

        # normalize column
        for j in range(rows):
            x[j][i] = (x[j][i] - min_value) / denominator

    print("\nNormalized X:\n", x)
    return x


def main():
    data = load_data()
    x, y = form_x_y(data)
    normalize(x)

if __name__ == "__main__":
    main()


"""Write a program to partition a dataset (simulated data for regression)  into two parts,
based on a feature (BP) and for a threshold, t = 80.
Generate additional two partitioned datasets based on different threshold values of t = [78, 82].
"""
import pandas as pd

def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    return data

def partition_data(data,threshold):
    value_lt_threshold = data[data["BP"] <= threshold ]                       #lt = less than
    value_gt_threshold = data[data["BP"]  > threshold ]                       #gt= greater than

    print("=" * 50)
    print(f"\nthreshold: {threshold}\n")
    print("=" * 50)
    
    print(value_lt_threshold.head())
    print(value_lt_threshold.shape)
    print(value_gt_threshold.head())
    print(value_gt_threshold.shape)


def main():
    data = load_data()

    threshold = [80,78,82]

    for threshold in threshold:
        partition_data(data,threshold)

if __name__ == "__main__":
    main()

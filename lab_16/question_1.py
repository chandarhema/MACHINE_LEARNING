"""Write a Python program to aggregate  predictions from
multiple trees to output a final prediction for a regression problem."""

import numpy as np

# Each row = predictions from all trees for one sample
# Shape: (n_samples, n_trees)
tree_predictions = np.array([
    [10.5, 11.0, 9.8, 10.2, 10.9],
    [20.1, 19.8, 20.5, 21.0, 20.3],
    [5.2, 5.5, 55.1, 5.3, 5.4]
])

def aggregate_predictions(predictions):
    return np.mean(predictions, axis=1)

final_predictions = aggregate_predictions(tree_predictions)

print("Final Predictions:", final_predictions)
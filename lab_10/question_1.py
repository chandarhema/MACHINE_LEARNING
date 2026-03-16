"""Implement entropy measure using Python.
The function should accept a set of data points and their class labels and return the entropy value.
"""

import math

def entropy(labels):
    # Step 1: count occurrences of each class
    counts = {}
    for label in labels:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    # Step 2: total number of samples
    total = len(labels)

    # Step 3: calculate entropy
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent

# Example dataset
labels = ["Yes", "No", "Yes", "Yes", "No", "No"]

print("Entropy:", entropy(labels))
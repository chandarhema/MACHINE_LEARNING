"""Implement information gain measures. The function should accept data points for parents,
data points for both children and return an information gain value."""
import math


# Entropy function
def entropy(labels):
    total = len(labels)
    ent = 0

    for label in set(labels):
        p = labels.count(label) / total
        ent -= p * math.log2(p)

    return ent


# Information Gain function
def information_gain(parent, child1, child2):
    # Entropy of parent
    parent_entropy = entropy(parent)

    # Sizes
    total = len(parent)
    w1 = len(child1) / total
    w2 = len(child2) / total

    # Weighted entropy of children
    children_entropy = w1 * entropy(child1) + w2 * entropy(child2)

    # Information Gain
    ig = parent_entropy - children_entropy

    return ig


# Example dataset
parent = ["Yes", "Yes", "No", "Yes", "No", "No"]
child1 = ["Yes", "Yes", "No"]
child2 = ["Yes", "No", "No"]

print("Information Gain:", information_gain(parent, child1, child2))

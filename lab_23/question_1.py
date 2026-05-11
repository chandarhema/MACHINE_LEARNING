"""
    Develop prediction model for Iris.csv using joint probability distribution approach
    Use only the first two features, SepalLengthCm, SepalWidthCm and the target variable
    Add random noise to the features
    Discretize the feature values
    Build a decision tree model with max_depth = 2, then,
    compare the accuracy of this model with the joint probability distribution method
"""
"""
Iris Classification using:

1. Joint Probability Distribution
2. Decision Tree Classifier

Conditions:
- Use only SepalLengthCm and SepalWidthCm
- Add random noise
- Discretize feature values
- Compare both model accuracies
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================================
# STEP 1 : READ DATASET
# ==========================================================

dataset = pd.read_csv("Iris.csv")

X = dataset[["SepalLengthCm", "SepalWidthCm"]]
y = dataset["Species"]

# ==========================================================
# STEP 2 : ADD RANDOM NOISE
# ==========================================================

np.random.seed(50)

noise_matrix = np.random.normal(
    0,
    0.2,
    X.shape
)

X_noisy = X + noise_matrix

# ==========================================================
# STEP 3 : DISCRETIZATION
# ==========================================================

discretizer = KBinsDiscretizer(
    n_bins=5,
    encode="ordinal",
    strategy="uniform",
    quantile_method="linear"
)

X_discrete = discretizer.fit_transform(X_noisy)

# ==========================================================
# STEP 4 : SPLIT TRAINING AND TESTING
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_discrete,
    y,
    test_size=0.30,
    random_state=5
)

# ==========================================================
# STEP 5 : CREATE JOINT PROBABILITY MODEL
# ==========================================================

probability_database = {}

for features, label in zip(X_train, y_train):

    key = tuple(features.astype(int))

    if key not in probability_database:
        probability_database[key] = {}

    if label not in probability_database[key]:
        probability_database[key][label] = 0

    probability_database[key][label] += 1

# Most common class fallback
default_class = y_train.mode()[0]

# ==========================================================
# STEP 6 : PREDICTION USING JOINT PROBABILITY
# ==========================================================

joint_predictions = []

for row in X_test:

    row_key = tuple(row.astype(int))

    if row_key in probability_database:

        label_counts = probability_database[row_key]

        predicted_class = max(
            label_counts,
            key=label_counts.get
        )

    else:
        predicted_class = default_class

    joint_predictions.append(predicted_class)

# ==========================================================
# STEP 7 : DECISION TREE MODEL
# ==========================================================

decision_tree = DecisionTreeClassifier(
    max_depth=2,
    random_state=5
)

decision_tree.fit(X_train, y_train)

tree_predictions = decision_tree.predict(X_test)

# ==========================================================
# STEP 8 : ACCURACY COMPARISON
# ==========================================================

joint_accuracy = accuracy_score(
    y_test,
    joint_predictions
)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

# ==========================================================
# STEP 9 : DISPLAY RESULTS
# ==========================================================

print("\n====================================")
print("JOINT PROBABILITY MODEL ACCURACY")
print("====================================")

print(
    round(joint_accuracy * 100, 2),
    "%"
)

print("\n====================================")
print("DECISION TREE MODEL ACCURACY")
print("====================================")

print(
    round(tree_accuracy * 100, 2),
    "%"
)

print("\n====================================")
print("COMPARISON OF FIRST 10 SAMPLES")
print("====================================")

for i in range(10):

    print("\nSample :", i + 1)

    print(
        "Actual Class           :",
        y_test.iloc[i]
    )

    print(
        "Joint Probability Pred :",
        joint_predictions[i]
    )

    print(
        "Decision Tree Pred     :",
        tree_predictions[i]
    )
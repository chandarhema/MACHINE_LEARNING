"""Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features.
Leave out 10% of each class and test prediction performance on these observations.
https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html#supervised-learning-tut
 - Check the solution code to learn about various plots."""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# Take only class 0 and 1
X = iris.data[:100, :2]
y = iris.target[:100]

# Split (10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# Train SVM
model = SVC(kernel='rbf', gamma=0.5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
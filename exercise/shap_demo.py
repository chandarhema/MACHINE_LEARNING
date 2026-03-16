import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('Autism-Adult-Data.arff')

# Encode categorical columns
categorical = ["gender", "jaundice", "family_history"]

le = LabelEncoder()

for col in categorical:
    data[col] = le.fit_transform(data[col])

# Target
y = data["ASD"]

# Features
X = data.drop("ASD", axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# SHAP explanation
explainer = shap.Explainer(model, X_train)

shap_values = explainer(X_test)

# SHAP plot
shap.summary_plot(shap_values, X_test)
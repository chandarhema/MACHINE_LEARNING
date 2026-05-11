import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# 🔹 Load dataset
def load_data():
    df = pd.read_csv("Tweets.csv")
    return df[['text', 'airline_sentiment']]


# 🔹 Split features and target
def split_target(df):
    X = df['text']
    y = df['airline_sentiment']
    return X, y


# 🔹 Train-test split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


# 🔹 TF-IDF Vectorization
def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec


# 🔹 Train SVM model
def train_model(X_train, y_train, kernel_type):
    model = SVC(kernel=kernel_type)
    model.fit(X_train, y_train)
    return model


# 🔹 Evaluate model
def evaluate_model(model, X_test, y_test, kernel_name):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n--- SVM ({kernel_name}) ---")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))


# 🔹 Run full pipeline
def run_svm(X_train, X_test, y_train, y_test):
    # Linear
    model_linear = train_model(X_train, y_train, 'linear')
    evaluate_model(model_linear, X_test, y_test, 'linear')

    # RBF
    model_rbf = train_model(X_train, y_train, 'rbf')
    evaluate_model(model_rbf, X_test, y_test, 'rbf')

    # Polynomial
    model_poly = train_model(X_train, y_train, 'poly')
    evaluate_model(model_poly, X_test, y_test, 'poly')


# 🔹 Main function
def main():
    # Step 1: Load data
    df = load_data()

    # Step 2: Split X and y
    X, y = split_target(df)

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Convert text → vectors
    X_train_vec, X_test_vec = vectorize(X_train, X_test)

    # Step 5: Train + evaluate SVM
    run_svm(X_train_vec, X_test_vec, y_train, y_test)


if __name__ == "__main__":
    main()
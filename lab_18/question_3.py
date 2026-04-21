from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# -------------------------------
# DATASET (balanced, sufficient)
# -------------------------------

texts = [
    # Positive (1)
    "good movie", "great film", "awesome acting", "loved it",
    "excellent story", "fantastic plot", "superb direction",
    "nice screenplay", "amazing performance", "brilliant movie",
    "really enjoyed it", "best movie ever", "so good",
    "liked the acting", "great experience",

    # Negative (0)
    "bad movie", "worst film", "boring plot", "hate it",
    "terrible acting", "awful story", "poor direction",
    "waste of time", "not good", "very bad movie",
    "did not like it", "very boring", "so bad",
    "disappointing film", "worst experience"
]

labels = [1] * 15 + [0] * 15  # 1 = positive, 0 = negative

# -------------------------------
# TEXT → NUMERICAL FEATURES
# -------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)

# -------------------------------
# FEATURE SCALING (important for SVM)
# -------------------------------

scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, stratify=labels, random_state=42
)

# -------------------------------
# TRAIN + TEST DIFFERENT KERNELS
# -------------------------------

for kernel in ['linear', 'rbf', 'poly']:
    model = SVC(kernel=kernel, class_weight='balanced')
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nKernel:", kernel)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Predicted:", pred)
    print("Actual:   ", y_test)
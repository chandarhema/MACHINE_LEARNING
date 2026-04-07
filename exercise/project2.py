import pandas as pd
import numpy as np

# =========================
# 1. EXTRACT LABELS FROM FILE
# =========================
def extract_labels(file_path):
    titles = []
    diagnosis = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                titles = line.strip().split("\t")[1:]
            if line.startswith("!Sample_characteristics_ch1"):
                diagnosis = line.strip().split("\t")[1:]
                break

    samples = [t.replace("blood sample_", "") for t in titles]

    labels = []
    for s, d in zip(samples, diagnosis):
        if "ASD" in d:
            label = "ASD"
        elif "TD" in d:
            label = "Control"
        else:
            label = "Unknown"

        labels.append([s, label])

    df_labels = pd.DataFrame(labels, columns=["Sample", "Label"])
    df_labels = df_labels[df_labels["Label"] != "Unknown"]

    print("Labels extracted:", df_labels.shape)
    return df_labels


# =========================
# 2. LOAD + CLEAN DATA
# =========================
def load_and_clean(file_path):
    df = pd.read_csv(file_path, sep="\t")

    cols_to_keep = ["PROBE_ID"] + [col for col in df.columns if "AVG_Signal" in col]
    df_clean = df[cols_to_keep]

    df_clean.columns = [col.replace(".AVG_Signal", "") for col in df_clean.columns]
    df_clean.set_index("PROBE_ID", inplace=True)

    df_clean = df_clean.T

    print("Shape after cleaning:", df_clean.shape)
    return df_clean


# =========================
# 3. LOG TRANSFORM
# =========================
def log_transform(df):
    return np.log2(df + 1)


# =========================
# 4. ALIGN LABELS
# =========================
def prepare_data(df, labels_df):
    labels_df = labels_df.set_index("Sample")

    df = df.loc[labels_df.index]

    X = df
    y = labels_df["Label"].map({"Control": 0, "ASD": 1})

    return X, y


# =========================
# 5. SPLIT
# =========================
def split_data(X, y):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


# =========================
# 6. FEATURE SELECTION
# =========================
def feature_selection(X_train, X_test, y_train, k=1000):
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(score_func=f_classif, k=k)

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    return X_train_sel, X_test_sel


# =========================
# 7. SCALING
# =========================
def scale_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# =========================
# 8. MODEL
# =========================
def train_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)

    return model


# =========================
# 9. EVALUATE
# =========================
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))


# =========================
# 10. SHAP
# =========================
def explain_model(model, X_train, X_test):
    import shap

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test)


# =========================
# MAIN
# =========================
def run_pipeline(file_path):
    # Extract labels
    labels_df = extract_labels(file_path)

    # Load data
    df = load_and_clean(file_path)

    # Log transform
    df = log_transform(df)

    # Prepare X, y
    X, y = prepare_data(df, labels_df)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature selection
    X_train, X_test = feature_selection(X_train, X_test, y_train)

    # Scaling
    X_train, X_test = scale_data(X_train, X_test)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Explain
    explain_model(model, X_train, X_test)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_pipeline("GSE42133_non-normalized_data.txt")
# =========================
# 1. IMPORTS
# =========================
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# =========================
# 2. LOAD DATA
# =========================
def load_nci_data(filepath):
    nci = pd.read_csv(filepath)

    # Drop sample ID
    nci = nci.drop(columns=['Unnamed: 0'])

    # Features + labels
    X = nci.drop(columns=['labs']).values
    y = nci['labs'].values

    return X, y


# =========================
# 3. REMOVE RARE CLASSES
# =========================
def remove_rare_classes(X, y, min_samples=3):
    counts = Counter(y)

    valid_classes = [cls for cls, c in counts.items() if c >= min_samples]
    mask = np.isin(y, valid_classes)

    X_filtered = X[mask]
    y_filtered = y[mask]

    print("Remaining classes:", Counter(y_filtered))

    return X_filtered, y_filtered


# =========================
# 4. SCALE DATA
# =========================
def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# =========================
# 5. HIERARCHICAL CLUSTERING FEATURES
# =========================
def hierarchical_features(X_scaled, n_clusters=50):

    print("Running hierarchical clustering...")

    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average'
    )

    # Cluster genes (transpose)
    gene_labels = hc.fit_predict(X_scaled.T)

    X_new = np.zeros((X_scaled.shape[0], n_clusters))

    for k in range(n_clusters):
        idx = np.where(gene_labels == k)[0]

        if len(idx) > 0:
            X_new[:, k] = X_scaled[:, idx].mean(axis=1)

    return X_new


# =========================
# 6. PCA FEATURES
# =========================
def pca_features(X_scaled, n_components=50):

    print("Running PCA...")

    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_scaled)


# =========================
# 7. EVALUATION (CROSS-VALIDATION)
# =========================
def evaluate_model(X, y):

    clf = SVC(kernel='linear')

    scores = cross_val_score(clf, X, y, cv=5)

    return scores.mean()


# =========================
# 8. MAIN PIPELINE
# =========================
def run_pipeline():

    # Load
    X, y = load_nci_data("NCI60.csv")
    print("Original shape:", X.shape)

    # Remove rare classes
    X, y = remove_rare_classes(X, y, min_samples=3)

    # Scale
    X_scaled = scale_data(X)

    # ---- Hierarchical Clustering ----
    X_hc = hierarchical_features(X_scaled, n_clusters=50)
    hc_score = evaluate_model(X_hc, y)

    # ---- PCA ----
    X_pca = pca_features(X_scaled, n_components=50)
    pca_score = evaluate_model(X_pca, y)

    # =========================
    # RESULTS
    # =========================
    print("\n=== FINAL RESULTS ===")
    print(f"Hierarchical Clustering CV Accuracy: {hc_score:.4f}")
    print(f"PCA CV Accuracy: {pca_score:.4f}")

    print("\n=== CONCLUSION ===")
    if pca_score > hc_score:
        print("PCA performs better")
    else:
        print("Hierarchical clustering performs comparably or better")


# =========================
# 9. RUN
# =========================
if __name__ == "__main__":
    run_pipeline()
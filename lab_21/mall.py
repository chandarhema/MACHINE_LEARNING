import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    data = df.drop(columns=['CustomerID'])
    data = pd.get_dummies(data, drop_first=True)
    return data

def eda(data):
    print("DESCRIPTION OF THE DATA")
    print(data.describe())
    print("\nHEAD OF THE DATA")
    print(data.head())
    print("\nMISSING VALUES")
    print(data.isnull().sum())

def preprocess(data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    return X_scaled, scaler

def apply_kmeans(X_scaled, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    return labels, centroids, kmeans

def reduce_dimension(X_scaled, centroids):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(centroids)
    return X_pca, centroids_pca

def compute_silhouette(X_scaled, labels):
    return silhouette_score(X_scaled, labels)

def plot_clusters(X_pca, labels, centroids_pca):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X')
    plt.title('K-Means Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

def main():
    # STEP 1: Load
    data = load_data()

    # STEP 2: EDA
    eda(data)

    # STEP 3: Preprocess
    X_scaled, scaler = preprocess(data)

    # STEP 4: KMeans
    k = 4
    labels, centroids, model = apply_kmeans(X_scaled, k)

    # STEP 5: Silhouette
    score = compute_silhouette(X_scaled, labels)
    print("Silhouette Score:", score)

    # STEP 6: PCA (only for visualization)
    X_pca, centroids_pca = reduce_dimension(X_scaled, centroids)

    # STEP 7: Plot
    plot_clusters(X_pca, labels, centroids_pca)

if __name__ == "__main__":
    main()
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# generate clean data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)

# scale
X = StandardScaler().fit_transform(X)

# kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# score
score = silhouette_score(X, labels)
print("Silhouette Score:", score)

# centroids
centroids = kmeans.cluster_centers_

# scatter plot
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X')
plt.title("K-Means Clustering (Blobs)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
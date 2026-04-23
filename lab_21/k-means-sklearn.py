#!/usr/bin/env python3

"""
KMeans with Pipeline (Scaling + PCA + Clustering + Silhouette Score)
Structured with main() for clarity
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances


def load_data():
    data = load_iris()
    return data.data


def build_pipeline(k,X):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('kmeans', KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            random_state=42
        ))
    ])
    pipeline=pipeline.fit(X)
    return pipeline

#
# def fit_pipeline(pipeline, X):
#     pipeline.fit(X)
#     return pipeline


def transform_data(pipeline, X):
    X_scaled = pipeline.named_steps['scaler'].transform(X)
    X_pca = pipeline.named_steps['pca'].transform(X_scaled)
    return X_pca


def get_kmeans_results(pipeline):
    kmeans = pipeline.named_steps['kmeans']
    # print(kmeans.cluster_centers_)
    # print(kmeans.labels_)
    return kmeans.cluster_centers_, kmeans.labels_


def compute_silhouette(X_pca, labels):
    return silhouette_score(X_pca, labels)


def plot_clusters(X, labels, centroids):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                color='red', marker='X', s=200, label='Centroids')

    plt.title("KMeans with PCA (Pipeline)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

def compute_distances(X, centroids):
    """
    Compute distance between each point and centroid
    """
    distances = euclidean_distances(X, centroids)
    return distances


def main():
    k = 2

    # Step 1: Load data
    X = load_data()

    # Step 2: Build pipeline
    pipeline = build_pipeline(k,X)

    # # Step 3: Fit pipeline
    # pipeline = fit_pipeline(pipeline, X)

    # Step 4: Transform data
    X_pca = transform_data(pipeline, X)

    # Step 5: Get results
    centroids, labels = get_kmeans_results(pipeline)

    # Step 6: Evaluate
    score = compute_silhouette(X_pca, labels)

    print("Final Centroids (in PCA space):\n", centroids)
    print("Silhouette Score:", score)

    # Step 7: Plot
    plot_clusters(X_pca, labels, centroids)

    #step 8 :
    distances = compute_distances(X_pca, centroids)
    # print("Distances:\n", distances)


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

def load_dataset():
    X= load_iris()
    return X.data

def model(X,k):
    pipeline = Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA(n_components=2)),
                     ('kmeans', KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42))])
    pipeline.fit(X)
    return pipeline

def preprocessing(pipeline,X):
    X_scaled = pipeline.named_steps['scaler'].transform(X)
    X_pca = pipeline.named_steps['pca'].transform(X_scaled)
    return X_pca

def kmeans_algo(pipeline):
    kmeans = pipeline.named_steps['kmeans']
    centroid=kmeans.cluster_centers_
    labels=kmeans.labels_
    return centroid, labels

def compute_silhouette_score(X_pca, labels):
    silhouette = silhouette_score(X_pca, labels)
    return silhouette

def plotting(X,labels,centroid):
    plt.scatter(X[:,0], X[:,1], c=labels, cmap= 'viridis')
    plt.scatter(centroid[:,0], centroid[:,1], color='red', marker='X', label='cluster centers(or)Centroids')
    plt.title('PCA Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def main():
    # step1 : loading the data
    
    X=load_dataset()
    
    # step2 :no of clusters
    k=2
    
    # step3 :preprocessing of steps
    pipeline=model(X,k)
    
    # step 4 : 
    X_pca = preprocessing(pipeline,X)

    # step 5 :
    centroids,labels=kmeans_algo(pipeline)
    print("centroids:",centroids)
    # print("labels:",labels)
    
    # step 6 :
    silhoutte=compute_silhouette_score(X_pca,labels)
    print("silhouette_score: ", silhoutte)

    #step 7 :
    plotting(X_pca,labels,centroids)
    
    
if __name__ == '__main__':
    main()

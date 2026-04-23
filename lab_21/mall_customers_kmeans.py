import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    data = df.drop(columns=['CustomerID'])
    data = pd.get_dummies(data,drop_first=True)
    return data

def eda(data):
    print("DESCRIPTION OF THE DATA")
    print(data.describe())
    print("\nHEAD OF THE DATA")
    print(data.head())
    print("\nTAIL OF THE DATA")
    print(data.tail())
    print("\nCOLUMNS OF THE DATA")
    print(data.columns)
    print("\nSHAPE OF THE DATA")
    print(data.shape)
    print("MISSING VALUES OF THE DATA")
    print(data.isnull().sum())
    return data

def making_pipeline(data,k):
    pipeline = Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA(n_components=2)),
                     ('kmeans', KMeans(n_clusters=k,init='k-means++', n_init=10, max_iter=300, random_state=42))])
    pipeline.fit(data)
    return pipeline

def transform_data(pipeline,data):
    x_scaled = pipeline.named_steps['scaler'].transform(data)
    x_pca = pipeline.named_steps['pca'].transform(x_scaled)
    return x_pca

def kmeans_algorithm(pipeline):
    kmeans = pipeline.named_steps['kmeans']
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centroids, labels

def compute_silhouette_score(X_pca, labels):
    silhouette = silhouette_score(X_pca, labels)
    return silhouette

def plot_clusters(X_pca, labels, centroids):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1],color='red',marker='X',label='centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


def main():
    #STEP 1 : LOADING THE DATA
    data = load_data()

    #STEP 2 :TO CHECK THE MISSING VALUES MAINLY
    data = eda(data)

    k=4

    #STEP 3 :CREATING THE PIPELINE
    pipeline = making_pipeline(data,k)

    #STEP 4 : transforming the data
    x_pca = transform_data(pipeline,data)

    #STEP 5 : centroids, labels
    centroids, labels = kmeans_algorithm(pipeline)

    #STEP 6 : COMPUTE THE SILHOUETTE SCORE
    silhouette = compute_silhouette_score(x_pca, labels)
    print("Silhouette Score: ", silhouette)

    plot_clusters(x_pca, labels, centroids)

if __name__ == "__main__":
    main()


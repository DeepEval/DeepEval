import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

def method():
    # Generate synthetic data
    n_samples = 300
    n_features = 2
    n_clusters = 4
    
    # Create sample data with blobs
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=0)
    
    # Standardize features
    X = StandardScaler().fit_transform(X)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_labels = kmeans.fit_predict(X)
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    # Plotting results
    plt.figure(figsize=(12, 5))

    # KMeans clustering results
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # DBSCAN clustering results
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

    # Return the labels for further analysis if needed
    output = {
        'kmeans_labels': kmeans_labels,
        'dbscan_labels': dbscan_labels
    }
    return output

# Call the method for validation
output = method()
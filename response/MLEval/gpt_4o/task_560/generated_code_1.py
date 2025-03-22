from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def method():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans_labels = kmeans.fit_predict(X)

    # Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=4)
    agglomerative_labels = agglomerative.fit_predict(X)

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # KMeans Plot
    axs[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
    axs[0].set_title('KMeans Clustering')

    # Agglomerative Plot
    axs[1].scatter(X[:, 0], X[:, 1], c=agglomerative_labels, cmap='viridis')
    axs[1].set_title('Agglomerative Clustering')

    # DBSCAN Plot
    axs[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
    axs[2].set_title('DBSCAN Clustering')

    plt.show()

    # For the purpose of this example, return KMeans labels
    output = kmeans_labels
    return output

# Call the method() for validation
output_labels = method()
print("Cluster labels from KMeans:", output_labels)
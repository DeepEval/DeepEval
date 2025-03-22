import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def method():
    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Evaluate KMeans clustering
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    print(f"KMeans Silhouette Score: {kmeans_silhouette}")

    # Apply Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg_clustering.fit_predict(X)

    # Evaluate Agglomerative Clustering
    agg_silhouette = silhouette_score(X, agg_labels)
    print(f"Agglomerative Clustering Silhouette Score: {agg_silhouette}")

    # Determine the best clustering method based on silhouette score
    if kmeans_silhouette > agg_silhouette:
        final_labels = kmeans_labels
    else:
        final_labels = agg_labels

    # Prepare the output
    output = {
        "KMeans Labels": kmeans_labels,
        "Agglomerative Labels": agg_labels,
        "Final Labels": final_labels
    }

    return output

# Call the method for validation
output = method()
print(output)
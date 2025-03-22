import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

def method():
    # Generate synthetic data (for example purposes, replace with your dataset)
    # X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    # For actual use, replace the above line with loading your dataset, e.g.:
    # data = pd.read_csv('your_dataset.csv')
    # X = data[['feature1', 'feature2']].values  # Update with appropriate feature names

    # For demonstration purposes, we are generating random data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

    # Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applying KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X_scaled)

    # Getting the cluster labels
    labels = kmeans.labels_

    # Optional: Visualizing the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('Consumer Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Return the labels and centers for further analysis if needed
    output = {
        "labels": labels,
        "centers": kmeans.cluster_centers_
    }
    return output

# Call the method for validation
result = method()
print(result)
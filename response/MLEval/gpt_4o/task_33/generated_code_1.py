import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def method(data, n_clusters):
    # Initializing KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fitting the model
    kmeans.fit(data)
    
    # Getting the centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Output can be the centroids, labels, or both
    output = (centroids, labels)
    
    return output

def validate_method():
    # Create a sample dataset
    np.random.seed(42)
    data = np.random.rand(100, 2)
    
    # Specify the number of clusters
    n_clusters = 3
    
    # Call the method
    centroids, labels = method(data, n_clusters)
    
    # Plot the results
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Call the validate function
validate_method()
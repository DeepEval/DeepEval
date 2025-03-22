from sklearn.cluster import KMeans
import numpy as np

def method():
    # Let's assume we have some sample data in a NumPy array
    # For demonstration purposes, we'll create random data
    np.random.seed(0)
    data = np.random.rand(100, 2)  # 100 samples, 2 features

    # Set the number of clusters k=30
    k = 30
    
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    
    # Get the cluster centers or labels
    output = kmeans.labels_  # or kmeans.cluster_centers_ if you want the centers

    return output

# Call the method for validation
output = method()
print(output)
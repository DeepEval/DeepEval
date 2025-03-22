from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def method():

    # Create a KMeans object with the desired number of clusters
    kmeans = KMeans(n_clusters=3)

    # Scale the data for better performance
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Fit the KMeans model to the scaled data
    kmeans.fit(data_scaled)

    # Obtain the cluster labels for each data point
    cluster_labels = kmeans.labels_

    # Return the cluster labels
    return cluster_labels

# Call the method() function for validation
cluster_labels = method()

# Print the cluster labels
print(cluster_labels)
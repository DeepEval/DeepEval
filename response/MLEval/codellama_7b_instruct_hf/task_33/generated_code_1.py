import numpy as np

def k_means(data, k):
    # Initialize the centroids randomly
    centroids = data[np.random.choice(len(data), k, replace=False)]

    # Loop until the centroids converge
    while True:
        # Assign the data points to the nearest centroid
        assignments = np.argmin(np.sum((data - centroids)**2, axis=1), axis=1)

        # Update the centroids
        centroids = np.mean(data[assignments == i], axis=0) for i in range(k)

        # Check if the centroids have converged
        if np.allclose(centroids, centroids_prev):
            break

    return centroids

# Example usage
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
k = 2
centroids = k_means(data, k)
print(centroids)
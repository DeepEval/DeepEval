from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
n_samples = 1000
centers = [(-5, 5), (5, 5), (-5, -5)]
X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=0)

# Fit the K-Means model
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()

# Print the cluster centers
print(kmeans.cluster_centers_)
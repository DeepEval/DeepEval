# Import necessary packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Generate some sample data (replace with your actual data)
np.random.seed(0)
data = {
    'Age': np.random.randint(18, 80, 100),
    'Income': np.random.randint(30000, 150000, 100),
    'Spendings': np.random.randint(5000, 20000, 100)
}
df = pd.DataFrame(data)

def method():
    """
    This function groups consumers into three clusters using K-means clustering algorithm.
    
    Returns:
    output (dict): The centroids of the three clusters and the labels of each consumer.
    """
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['Age', 'Income', 'Spendings']])
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(scaled_data)
    
    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_
    
    # Get the labels of each consumer
    labels = kmeans.labels_
    
    # Store the output in a dictionary
    output = {
        'centroids': centroids,
        'labels': labels
    }
    
    return output

# Call the method for validation
output = method()
print(output)
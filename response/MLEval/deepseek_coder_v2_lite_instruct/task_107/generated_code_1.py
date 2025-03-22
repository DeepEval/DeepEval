import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def method():
    # Example data: replace this with your actual dataset
    data = {
        'Feature1': [2.2, 3.4, 4.5, 5.1, 6.3, 7.2, 8.1, 9.0, 10.1, 11.2],
        'Feature2': [1.1, 2.3, 3.2, 4.1, 5.2, 6.1, 7.0, 8.1, 9.2, 10.3],
        'Feature3': [0.5, 1.2, 2.1, 3.3, 4.2, 5.1, 6.0, 7.1, 8.2, 9.3]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Initialize the KMeans model with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    
    # Fit the model to the data
    kmeans.fit(df)
    
    # Get the cluster labels
    labels = kmeans.labels_
    
    # Add the cluster labels to the DataFrame
    df['Cluster'] = labels
    
    # For demonstration, let's return the cluster labels
    output = df[['Cluster']]
    
    return output

# Call the method for validation
output = method()
print(output)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def method():
    # Sample consumer data for demonstration purposes.
    # In practice, you would replace this with your actual consumer data.
    data = {
        'Feature1': [1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5],
        'Feature2': [1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    # Define the KMeans model
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Fit the model and predict the clusters
    clusters = kmeans.fit_predict(scaled_features)

    # Add the cluster information to the original DataFrame
    df['Cluster'] = clusters

    # Plot the clusters
    plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis')
    plt.title('Consumer Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Return the DataFrame with cluster information as output
    output = df
    return output

# Call the method for validation
result = method()
print(result)
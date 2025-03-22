import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def method():
    # Load your data into a DataFrame
    # Replace 'data.csv' with your actual dataset
    # data = pd.read_csv('data.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_107.csv')
    
    # Assuming the dataset has two columns, 'Feature1' and 'Feature2'
    X = data[['Feature1', 'Feature2']]  # Replace with your actual column names
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Number of clusters
    n_clusters = 3
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # Get the cluster labels
    labels = kmeans.labels_
    
    # Calculate the silhouette score
    score = silhouette_score(X_scaled, labels)
    
    # Plot the clustering results
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
    plt.title(f'Silhouette Score: {score}')
    plt.show()
    
    # Return the cluster labels
    return labels

# Call the method for validation
output = method()
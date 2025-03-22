import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def method():
    # Load your dataset here. Replace 'your_dataset.csv' with the actual file path
    # data = pd.read_csv('your_dataset.csv') 

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_560.csv')

    # Select the features you want to use for clustering
    features = data[['feature1', 'feature2', 'feature3']]  # Replace with your actual feature columns

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
    kmeans.fit(scaled_features)

    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_

    # Add the cluster labels to the original DataFrame
    data['cluster'] = cluster_labels

    # Output the DataFrame with cluster assignments
    return data

# Call the method and print the output
output = method()
print(output)
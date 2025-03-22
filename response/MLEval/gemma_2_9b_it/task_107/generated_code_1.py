import pandas as pd
from sklearn.cluster import KMeans

def method():
    # Load your dataset 
    # Replace 'your_dataset.csv' with the actual path to your dataset
    # df = pd.read_csv('your_dataset.csv')

    # 修改为本地数据文件
    df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_107.csv')

    # Select the features you want to use for clustering
    # Replace 'feature1', 'feature2', etc. with the names of your features
    features = df[['feature1', 'feature2', 'feature3']] 

    # Create a KMeans model with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Fit the model to your data
    kmeans.fit(features)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Add the cluster labels to your dataframe
    df['cluster'] = labels

    # You can now analyze the clusters and the data points within each cluster
    # For example, you can calculate the mean of each feature for each cluster

    return df

# Call the method and print the output
output = method()
print(output)
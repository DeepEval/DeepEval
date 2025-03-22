# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def method():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Split the data into features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_train)
    kmeans_labels = kmeans.predict(X_train)
    print("KMeans Silhouette Score:", silhouette_score(X_train, kmeans_labels))

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=10, random_state=42)
    dbscan.fit(X_train)
    dbscan_labels = dbscan.labels_
    print("DBSCAN Silhouette Score:", silhouette_score(X_train, dbscan_labels))

    # Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    agglomerative.fit(X_train)
    agglomerative_labels = agglomerative.labels_
    print("Agglomerative Silhouette Score:", silhouette_score(X_train, agglomerative_labels))

    # Birch Clustering
    birch = Birch(n_clusters=3, threshold=1.0, random_state=42)
    birch.fit(X_train)
    birch_labels = birch.labels_
    print("Birch Silhouette Score:", silhouette_score(X_train, birch_labels))

    return kmeans_labels, dbscan_labels, agglomerative_labels, birch_labels

# Call the method for validation
kmeans_labels, dbscan_labels, agglomerative_labels, birch_labels = method()
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None, batch_reduction=True, point_reduction=True, norm=True, single_directional=True, abs_cosine=False):
    """
    Calculate the Chamfer distance and cosine distance of normals between two point clouds.

    Parameters:
    x (numpy array): First point cloud with shape (n, 3) or (n, 3+c) where c is the number of features.
    y (numpy array): Second point cloud with shape (m, 3) or (m, 3+c) where c is the number of features.
    x_lengths (numpy array, optional): Lengths of each point in the first point cloud.
    y_lengths (numpy array, optional): Lengths of each point in the second point cloud.
    x_normals (numpy array, optional): Normals of each point in the first point cloud.
    y_normals (numpy array, optional): Normals of each point in the second point cloud.
    weights (numpy array, optional): Weights for each point in the first point cloud.
    batch_reduction (bool, optional): Whether to reduce the results batch-wise.
    point_reduction (bool, optional): Whether to reduce the results point-wise.
    norm (bool, optional): Whether to compute the cosine distance of normals.
    single_directional (bool, optional): Whether to compute the distance in a single direction.
    abs_cosine (bool, optional): Whether to return the absolute cosine distance.

    Returns:
    tuple: A tuple containing the reduced Chamfer distance and cosine distance of normals.
    """
    # Normalize the point clouds and their normals
    x_norm = PCA(n_components=3).fit_transform(x)
    y_norm = PCA(n_components=3).fit_transform(y)
    x_normalized = x_norm / np.linalg.norm(x_norm, axis=1, keepdims=True)
    y_normalized = y_norm / np.linalg.norm(y_norm, axis=1, keepdims=True)

    # Calculate the Chamfer distance
    if x_lengths is not None and y_lengths is not None:
        tree_x = BallTree(x_normalized, leaf_size=10)
        tree_y = BallTree(y_normalized, leaf_size=10)
        distances_x, _ = tree_x.query(y_normalized, k=np.sum(x_lengths), return_distance=True, worker=0)
        distances_y, _ = tree_y.query(x_normalized, k=np.sum(y_lengths), return_distance=True, worker=0)
    else:
        distances_x, _ = tree_x.query(y_normalized, k=1, return_distance=True, worker=0)
        distances_y, _ = tree_y.query(x_normalized, k=1, return_distance=True, worker=0)

    # Calculate the cosine distance of normals
    cos_dist = pairwise_distances(y_normalized, x_normalized, metric='cosine')

    # Optionally reduce the results
    if batch_reduction:
        distances_x = np.mean(distances_x)
        distances_y = np.mean(distances_y)

    # Return the results
    if abs_cosine:
        return distances_x, distances_y, cos_dist
    else:
        return distances_x + distances_y, cos_dist

if __name__ == "__main__":
    # Sample input values
    x = np.random.random((100, 3))  # Random point cloud
    y = np.random.random((100, 3))  # Random point cloud
    lengths_x = np.random.random(100)  # Random lengths for x
    lengths_y = np.random.random(100)  # Random lengths for y
    normals_x = np.random.random((100, 3))  # Random normals for x
    normals_y = np.random.random((100, 3))  # Random normals for y
    weights = np.random.random(100)  # Random weights for x

    # Call the function and print the results
    distance, cosine = chamfer_distance(x, y, x_lengths=lengths_x, y_lengths=lengths_y, x_normals=normals_x, y_normals=normals_y, weights=weights, abs_cosine=True)
    print("Chamfer Distance:", distance)
    print("Cosine Distance:", cosine)
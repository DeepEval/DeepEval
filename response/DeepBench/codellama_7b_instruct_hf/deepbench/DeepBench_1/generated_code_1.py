import numpy as np
from scipy.spatial.distance import cdist

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None, batch_reduction='mean', point_reduction='mean', norm='l2', single_directional=False, abs_cosine=False):
    # Calculate the Euclidean distance between the point clouds
    distances = cdist(x, y, metric='euclidean')
    
    # Calculate the cosine distance between the point clouds
    cosine_distances = cdist(x, y, metric='cosine')
    
    # Calculate the reduced distance between the point clouds
    reduced_distance = distances
    if batch_reduction == 'mean':
        reduced_distance = np.mean(distances, axis=0)
    elif batch_reduction == 'sum':
        reduced_distance = np.sum(distances, axis=0)
    elif batch_reduction == 'max':
        reduced_distance = np.max(distances, axis=0)
    
    # Calculate the reduced cosine distance between the point clouds
    reduced_cosine_distance = cosine_distances
    if point_reduction == 'mean':
        reduced_cosine_distance = np.mean(cosine_distances, axis=1)
    elif point_reduction == 'sum':
        reduced_cosine_distance = np.sum(cosine_distances, axis=1)
    elif point_reduction == 'max':
        reduced_cosine_distance = np.max(cosine_distances, axis=1)
    
    # Calculate the absolute cosine distance between the point clouds
    if abs_cosine:
        reduced_cosine_distance = np.abs(reduced_cosine_distance)
    
    return reduced_distance, reduced_cosine_distance

if __name__ == "__main__":
    # Create sample input values
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Calculate the Chamfer distance between the point clouds
    distance, cosine_distance = chamfer_distance(x, y)
    
    # Print the results
    print(f"Chamfer distance: {distance}")
    print(f"Cosine distance: {cosine_distance}")
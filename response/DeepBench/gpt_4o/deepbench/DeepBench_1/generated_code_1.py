import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, 
                     weights=None, batch_reduction='mean', point_reduction='mean', 
                     norm='l2', single_directional=False, abs_cosine=False):

    def calculate_distance(a, b, norm):
        if norm == 'l2':
            return np.linalg.norm(a - b, axis=-1)
        elif norm == 'l1':
            return np.sum(np.abs(a - b), axis=-1)
        else:
            raise ValueError("Unsupported norm type. Use 'l2' or 'l1'.")

    def calculate_cosine_distance(a, b, abs_cosine):
        dot_product = np.sum(a * b, axis=-1)
        if abs_cosine:
            dot_product = np.abs(dot_product)
        return 1 - dot_product

    batch_size = x.shape[0]
    distances = []
    cos_distances = []

    for b in range(batch_size):
        x_cloud = x[b][:x_lengths[b]] if x_lengths is not None else x[b]
        y_cloud = y[b][:y_lengths[b]] if y_lengths is not None else y[b]

        tree_x = cKDTree(x_cloud)
        tree_y = cKDTree(y_cloud)

        dist_x_to_y, idx_x_to_y = tree_x.query(y_cloud)
        dist_y_to_x, idx_y_to_x = tree_y.query(x_cloud)

        if not single_directional:
            dist_x_to_y, dist_y_to_x = np.minimum(dist_x_to_y, dist_y_to_x), np.minimum(dist_y_to_x, dist_x_to_y)
        
        if weights is not None:
            dist_x_to_y *= weights[b][:len(dist_x_to_y)]
            dist_y_to_x *= weights[b][:len(dist_y_to_x)]

        if point_reduction == 'mean':
            dist_x = np.mean(dist_x_to_y)
            dist_y = np.mean(dist_y_to_x)
        elif point_reduction == 'sum':
            dist_x = np.sum(dist_x_to_y)
            dist_y = np.sum(dist_y_to_x)
        else:
            raise ValueError("Unsupported point reduction. Use 'mean' or 'sum'.")

        if single_directional:
            total_dist = dist_x
        else:
            total_dist = (dist_x + dist_y) / 2.0

        distances.append(total_dist)

        if x_normals is not None and y_normals is not None:
            x_n = x_normals[b][:x_lengths[b]] if x_lengths is not None else x_normals[b]
            y_n = y_normals[b][:y_lengths[b]] if y_lengths is not None else y_normals[b]

            cos_dist_x = calculate_cosine_distance(x_n, y_n[idx_y_to_x], abs_cosine)
            cos_dist_y = calculate_cosine_distance(y_n, x_n[idx_x_to_y], abs_cosine)
            
            cos_dist = (np.mean(cos_dist_x) + np.mean(cos_dist_y)) / 2.0 if not single_directional else np.mean(cos_dist_x)
            cos_distances.append(cos_dist)

    if batch_reduction == 'mean':
        chamfer_dist = np.mean(distances)
        chamfer_cos_dist = np.mean(cos_distances) if x_normals is not None else None
    elif batch_reduction == 'sum':
        chamfer_dist = np.sum(distances)
        chamfer_cos_dist = np.sum(cos_distances) if x_normals is not None else None
    else:
        raise ValueError("Unsupported batch reduction. Use 'mean' or 'sum'.")

    return chamfer_dist, chamfer_cos_dist

if __name__ == "__main__":
    x = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    y = np.array([[[0, 0, 1], [1, 1, 2], [2, 2, 3]], [[0, 0, 0], [0, 1, 0], [1, 1, 0]]])
    
    chamfer_dist, chamfer_cos_dist = chamfer_distance(x, y)
    print("Chamfer Distance:", chamfer_dist)
    print("Chamfer Cosine Distance:", chamfer_cos_dist)
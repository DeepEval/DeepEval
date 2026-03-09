import numpy as np

def compute_fundamental_matrix(points1, points2, weights=None):
    # Points to be computed
    points1 = points1.astype(np.float32)
    points2 = points2.astype(np.float32)

    # Check if weights are provided and reshape them if necessary
    if weights is not None:
        weights = weights.astype(np.float32)
        if weights.shape[1] != points1.shape[1]:
            weights = np.expand_dims(weights, 1)

    # Check if points have the correct shape
    if points1.shape[1] < 8 or points2.shape[1] < 8:
        raise ValueError("Points must have at least 8 points each.")

    # Homogeneous coordinates
    points1_homogeneous = np.hstack([points1, np.ones((points1.shape[0], 1))])
    points2_homogeneous = np.hstack([points2, np.ones((points2.shape[0], 1))])

    # Compute the essential matrix
    essential_matrix = points2_homogeneous.dot(points1_homogeneous.T)
    essential_matrix[:2, :2] /= essential_matrix[:2, 2].reshape(-1, 1)
    essential_matrix[:2, 2] = -essential_matrix[:2, 2]

    # Compute the fundamental matrix
    fundamental_matrix = essential_matrix.T.dot(essential_matrix)
    fundamental_matrix /= np.linalg.det(fundamental_matrix)

    return fundamental_matrix


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'path/to/your/installation')
    import cv2
    import matplotlib.pyplot as plt

    # Create sample input values
    import tensorflow as tf
    import numpy as np
    B = 1
    N = 10
    points1 = tf.constant(np.random.rand(B, N, 2), dtype=tf.float32)
    points2 = tf.constant(np.random.rand(B, N, 2), dtype=tf.float32)
    weights = tf.constant(np.random.rand(B, N), dtype=tf.float32)

    # Call the function and print the results
    fundamental_matrix = compute_fundamental_matrix(points1, points2, weights)
    print("Computed Fundamental Matrix:\n", fundamental_matrix)

    # Visualize the fundamental matrix
    fundamental_matrix_array = fundamental_matrix.numpy()
    u, s, v = np.linalg.svd(fundamental_matrix_array)
    F_refined = v[-1].reshape(3, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(F_refined, interpolation='nearest')
    plt.show()
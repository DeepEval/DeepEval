import numpy as np

def apply_transformation(R, T, s, X):
    """
    Applies a similarity transformation to a given d-dimensional cloud X.

    Parameters:
    R (np.ndarray): A batch of orthonormal matrices of shape (minibatch, d, d).
    T (np.ndarray): A batch of translations of shape (minibatch, d).
    s (np.ndarray): A batch of scaling factors of shape (minibatch,).
    X (np.ndarray): A d-dimensional cloud of shape (minibatch, num_points, d).

    Returns:
    np.ndarray: The transformed cloud of shape (minibatch, num_points, d).
    """
    batch_size = R.shape[0]
    d = R.shape[1]
    num_points = X.shape[1]

    # Initialize the transformed cloud with the original shape
    X_transformed = np.zeros_like(X)

    for batch_idx in range(batch_size):
        R_batch = R[batch_idx]
        T_batch = T[batch_idx]
        s_batch = s[batch_idx]

        # Transform each point in the cloud
        for point_idx in range(num_points):
            # Apply rotation
            for i in range(d):
                for j in range(d):
                    X_transformed[batch_idx, point_idx, i] += R_batch[i, j] * X[batch_idx, point_idx, j]

            # Apply translation
            X_transformed[batch_idx, point_idx, :] += T_batch

            # Apply scaling
            X_transformed[batch_idx, point_idx, :] *= s_batch

    return X_transformed

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    import numpy as np

    # Generate random orthonormal matrices, translations, and scaling factors
    random.seed(0)
    R = np.random.rand(5, 3, 3)  # minibatch, d, d
    T = np.random.rand(5, 3)  # minibatch, d
    s = np.random.rand(5)  # minibatch

    X = np.random.rand(5, 10, 3)  # minibatch, num_points, d

    # Apply the transformation
    X_transformed = apply_transformation(R, T, s, X)

    # Plot the original and transformed clouds
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(X_transformed[0, :, 0], X_transformed[0, :, 1], c='r', label='Transformed')
    ax[0].scatter(X_transformed[1, :, 0], X_transformed[1, :, 1], c='g', label='Original')
    ax[0].set_title('Transformation 1')
    ax[1].scatter(X_transformed[2, :, 0], X_transformed[2, :, 1], c='b', label='Transformed')
    ax[1].scatter(X_transformed[3, :, 0], X_transformed[3, :, 1], c='y', label='Original')
    ax[1].set_title('Transformation 2')
    plt.show()
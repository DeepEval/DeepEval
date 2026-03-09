import numpy as np

def apply_similarity_transformation(R, T, s, X):
    minibatch, num_points, d = X.shape

    transformed_X = np.einsum('bij,bpj->bpi', R, X)  # Apply rotation
    transformed_X = s[:, np.newaxis, np.newaxis] * transformed_X  # Apply scaling
    transformed_X += T[:, np.newaxis, :]  # Apply translation

    return transformed_X

if __name__ == "__main__":
    np.random.seed(42)

    minibatch = 2
    num_points = 3
    d = 2

    R = np.array([
        [[1, 0], [0, 1]],  # Identity matrix for batch 1
        [[0, -1], [1, 0]]  # 90 degree rotation for batch 2
    ])

    T = np.array([
        [1, 2],  # Translation for batch 1
        [3, 4]   # Translation for batch 2
    ])

    s = np.array([2, 0.5])  # Scaling factors for each batch

    X = np.array([
        [[1, 0], [0, 1], [1, 1]],  # Point cloud for batch 1
        [[1, 0], [0, 1], [1, 1]]   # Point cloud for batch 2
    ])

    transformed_X = apply_similarity_transformation(R, T, s, X)
    print("Transformed X:")
    print(transformed_X)
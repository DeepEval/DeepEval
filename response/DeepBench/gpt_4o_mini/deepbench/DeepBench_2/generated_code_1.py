import numpy as np

def apply_transformation(X, R, T, s):
    return s[:, np.newaxis, np.newaxis] * np.einsum('bik,bjk->bij', R, X) + T[:, np.newaxis, :]

if __name__ == "__main__":
    minibatch = 2
    num_points = 3
    d = 2

    X = np.random.rand(minibatch, num_points, d)  # Random cloud of points
    R = np.array([
        [[1, 0], [0, 1]],  # Identity matrix for the first batch
        [[0, -1], [1, 0]]  # 90-degree rotation for the second batch
    ])
    T = np.array([[1, 1], [2, 2]])  # Translations
    s = np.array([0.5, 1.5])  # Scaling factors

    transformed_X = apply_transformation(X, R, T, s)
    print("Original Points:\n", X)
    print("Transformed Points:\n", transformed_X)
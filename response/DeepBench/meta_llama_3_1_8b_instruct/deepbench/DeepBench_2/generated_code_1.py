import numpy as np

def apply_transformation(X, R, T, s):
    batch_size, num_points, d = X.shape
    for i in range(batch_size):
        X[i] = np.dot(R[i], X[i].T).T * s[i] + T[i]
    return X

if __name__ == "__main__":
    # Create sample input values
    np.random.seed(0)
    minibatch = 2
    d = 3
    num_points = 10
    R = np.random.rand(minibatch, d, d)
    np.fill_diagonal(R, 1)  # Ensure R is orthonormal
    R = R @ R.T  # Compute orthogonal matrix
    T = np.random.rand(minibatch, d)
    s = np.random.rand(minibatch)
    X = np.random.rand(minibatch, num_points, d)

    # Call the function and print the results
    transformed_X = apply_transformation(X, R, T, s)
    print(transformed_X)
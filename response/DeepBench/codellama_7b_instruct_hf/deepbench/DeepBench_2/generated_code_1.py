import numpy as np

def apply_similarity_transformation(R, T, s, X):
    # Reshape the inputs to matrices
    R = R.reshape(R.shape[0], R.shape[1], R.shape[2])
    T = T.reshape(T.shape[0], T.shape[1], 1)
    s = s.reshape(s.shape[0], 1, 1)

    # Compute the similarity transformation
    X_transformed = np.matmul(R, X.transpose(0, 2, 1)).transpose(0, 2, 1) + T
    X_transformed = X_transformed * s

    return X_transformed

if __name__ == "__main__":
    # Generate sample input values
    R = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    T = np.array([[0, 0, 0]])
    s = np.array([[1]])
    X = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

    # Call the function and print the results
    X_transformed = apply_similarity_transformation(R, T, s, X)
    print(X_transformed)
import numpy as np

def compute_eigenvalues_eigenvectors(X):
    eigen_vals, eigen_vecs = np.linalg.eig(X)
    return eigen_vals, eigen_vecs


X = np.array([[1, 2], [3, 4]])
eigen_vals, eigen_vecs = compute_eigenvalues_eigenvectors(X)
print(eigen_vals)
print(eigen_vecs)
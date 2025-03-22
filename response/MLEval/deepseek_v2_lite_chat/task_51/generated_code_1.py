import numpy as np

def method(X):
    """
    Compute the eigenvalues and eigenvectors of a square matrix X.

    Args:
    X (numpy.ndarray): A square matrix with real number entries.

    Returns:
    tuple: A tuple containing the eigenvalues and the eigenvectors.
    """
    # Compute the eigenvalues and eigenvectors of X
    eigenvalues, eigenvectors = np.linalg.eig(X)
    
    # Return the eigenvalues and eigenvectors
    return eigenvalues, eigenvectors

# Example usage of the method function
if __name__ == "__main__":
    # Define a sample matrix
    X = np.array([[1, 2], [3, 4]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = method(X)
    
    # Print the eigenvalues
    print("Eigenvalues:")
    print(eigenvalues)
    
    # Print the eigenvectors
    print("Eigenvectors:")
    print(eigenvectors)
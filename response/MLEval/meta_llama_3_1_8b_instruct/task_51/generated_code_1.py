import numpy as np

def method():
    """
    Compute the eigenvalues and eigenvectors of X.

    Returns:
        output (dict): A dictionary containing the eigenvalues and eigenvectors of X.
    """

    # Define a sample matrix X
    X = np.array([[2, 1], [1, 2]])

    # Compute the eigenvalues and eigenvectors of X
    eigenvalues, eigenvectors = np.linalg.eig(X)

    # Return the eigenvalues and eigenvectors as a dictionary
    return {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}

# Call the method for validation
output = method()

# Print the eigenvalues and eigenvectors
print("Eigenvalues:")
print(output['eigenvalues'])
print("\nEigenvectors:")
print(output['eigenvectors'])
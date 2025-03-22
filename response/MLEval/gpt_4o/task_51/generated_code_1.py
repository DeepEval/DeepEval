import numpy as np

def method():
    # Example matrix X
    X = np.array([[4, -2],
                  [1, 1]])

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(X)

    # Prepare the output
    output = (eigenvalues, eigenvectors)
    
    return output

# Call the method for validation
eigenvalues, eigenvectors = method()
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
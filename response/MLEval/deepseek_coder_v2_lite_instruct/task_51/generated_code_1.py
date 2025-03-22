import numpy as np

def method():
    # Example matrix X
    X = np.array([[4, -2], [-2, 4]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(X)
    
    # Prepare the output
    output = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }
    
    return output

# Call the method for validation
output = method()
print("Eigenvalues:", output['eigenvalues'])
print("Eigenvectors:", output['eigenvectors'])
import numpy as np

def method():
    # Define the matrix X (you can replace this with your specific matrix)
    X = np.array([[4, -2],
                  [1, 1]])

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(X)

    # Prepare the output
    output = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }

    return output

# Call the method and print the output for validation
result = method()
print(result)
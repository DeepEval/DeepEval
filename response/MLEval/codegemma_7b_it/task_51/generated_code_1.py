import numpy as np
import matplotlib.pyplot as plt

def method():

    # Generate a random 3x3 matrix
    X = np.random.rand(3, 3)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(X)

    # Print the eigenvalues and eigenvectors
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)

    # Plot the eigenvectors as vectors in a 2D plot
    fig, ax = plt.subplots()
    ax.quiver(0, 0, eigenvectors[0, 0], eigenvectors[0, 1], color='r', label='Eigenvector 1')
    ax.quiver(0, 0, eigenvectors[1, 0], eigenvectors[1, 1], color='g', label='Eigenvector 2')
    ax.quiver(0, 0, eigenvectors[2, 0], eigenvectors[2, 1], color='b', label='Eigenvector 3')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.legend()
    plt.show()

    # Return the eigenvalues and eigenvectors
    return eigenvalues, eigenvectors


# Call the method for validation
eigenvalues, eigenvectors = method()

# Print the output
print("Output:", eigenvalues, eigenvectors)
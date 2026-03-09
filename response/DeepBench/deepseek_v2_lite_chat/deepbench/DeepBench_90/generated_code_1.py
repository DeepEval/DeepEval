import numpy as np

def noise_matrix_is_valid(noise_matrix, target_matrix, verbose=False):
    """
    Checks if a given noise matrix is learnable, meaning it has a structure
    that allows for better than random performance improvement through learning.
    
    Args:
    noise_matrix (np.ndarray): The noise matrix to be checked.
    target_matrix (np.ndarray): The target matrix associated with the noise matrix.
    verbose (bool): If True, print additional information during the check.
    
    Returns:
    bool: True if the noise matrix is learnable, False otherwise.
    """
    # Check if the dimensions of the noise and target matrices are compatible
    if noise_matrix.shape != target_matrix.shape:
        raise ValueError("The noise matrix and target matrix must have the same shape.")
    
    # Calculate the number of unique elements in the target matrix
    num_unique_elements = len(np.unique(target_matrix))
    
    # Check if the noise matrix is a permutation of the identity matrix
    # A permutation of the identity matrix means each row and column has a unique element
    # This indicates that the noise matrix does not contribute to distinguishing between elements
    if np.all(noise_matrix == np.eye(noise_matrix.shape[0])):
        return False
    
    # Check if the noise matrix has uniform distribution
    if np.all(noise_matrix == np.ones(noise_matrix.shape) / noise_matrix.shape[0]):
        return False
    
    # Check if the noise matrix has a very low entropy
    # Lower entropy means higher predictability and thus less learnability
    if np.sum(np.abs(np.log2(noise_matrix + 1e-10))) < 1:
        return False
    
    # If none of the above conditions are met, the matrix is likely learnable
    return True

if __name__ == "__main__":
    # Create a sample input matrices
    np.random.seed(42)
    noise_matrix = np.random.randint(2, size=(3, 3))
    target_matrix = np.array([[0, 1], [1, 0]])
    
    # Check if the noise matrix is learnable
    is_learnable = noise_matrix_is_valid(noise_matrix, target_matrix)
    
    if verbose:
        print("Noise Matrix:")
        print(noise_matrix)
        print("Target Matrix:")
        print(target_matrix)
        print("Is the noise matrix learnable?", is_learnable)

    # Output should indicate whether the noise matrix is learnable
    # The output can be True or False, depending on the structure of the matrices
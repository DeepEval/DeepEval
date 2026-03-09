import numpy as np

def noise_matrix_is_valid(noise_matrix: np.ndarray, py: np.ndarray, verbose: bool = False) -> bool:
    """
    Check if the given noise_matrix is a learnable matrix.
    
    Parameters:
    - noise_matrix (np.ndarray): A matrix representing the noise in the system.
    - py (np.ndarray): A matrix of probabilities corresponding to the outputs.
    - verbose (bool): Flag to print detailed output information.

    Returns:
    - bool: True if the noise matrix is learnable, False otherwise.
    """
    # Check if the shapes of the input matrices match
    if noise_matrix.shape != py.shape:
        raise ValueError("noise_matrix and py must have the same shape.")

    # Calculate the total number of samples
    total_samples = np.sum(py)
    
    if total_samples == 0:
        if verbose:
            print("Total samples in py is zero, returning False.")
        return False
    
    # Calculate the expected output probabilities based on the noise matrix
    expected_output = np.dot(noise_matrix, py) / total_samples

    # Check if expected outputs are learnable (greater than random performance)
    random_performance = 1 / len(py)
    
    if np.any(expected_output > random_performance):
        if verbose:
            print("The noise matrix is learnable.")
        return True
    else:
        if verbose:
            print("The noise matrix is not learnable.")
        return False

if __name__ == "__main__":
    # Create sample noise matrix and probability matrix
    noise_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    py = np.array([1, 1])  # Example probabilities

    # Call the function and print the result
    result = noise_matrix_is_valid(noise_matrix, py, verbose=True)
    print(f"Is the noise matrix valid? {result}")
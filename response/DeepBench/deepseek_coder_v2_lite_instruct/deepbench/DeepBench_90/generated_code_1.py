import numpy as np

def noise_matrix_is_valid(noise_matrix, py, verbose=False):
    """
    Check if the given noise matrix is learnable.

    Parameters:
    noise_matrix (np.ndarray): A 2D numpy array representing the noise matrix.
    py (np.ndarray): A 1D numpy array representing the prior probabilities of the true labels.
    verbose (bool, optional): If True, print detailed information about the calculations.

    Returns:
    bool: True if the noise matrix is learnable, False otherwise.
    """
    # Check if the noise matrix is square
    if noise_matrix.shape[0] != noise_matrix.shape[1]:
        if verbose:
            print("Noise matrix is not square.")
        return False

    # Check if the noise matrix rows sum to 1
    if not np.allclose(noise_matrix.sum(axis=1), np.ones(noise_matrix.shape[0])):
        if verbose:
            print("Rows of noise matrix do not sum to 1.")
        return False

    # Check if the prior probabilities array is valid
    if py.ndim != 1 or py.shape[0] != noise_matrix.shape[0]:
        if verbose:
            print("Prior probabilities array is not valid.")
        return False

    # Calculate the probability of observing each noisy label
    px_given_y = noise_matrix.dot(py)

    # Check if all probabilities are greater than 0
    if np.any(px_given_y <= 0):
        if verbose:
            print("Some probabilities are less than or equal to 0.")
        return False

    # Calculate the mutual information
    mutual_info = (px_given_y * (np.log2(px_given_y) - np.log2(px_given_y.mean()))).sum()

    # Check if mutual information is positive
    if mutual_info <= 0:
        if verbose:
            print("Mutual information is less than or equal to 0.")
        return False

    if verbose:
        print("Noise matrix is learnable.")

    return True

if __name__ == "__main__":
    # Sample input values
    noise_matrix = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.1, 0.7]])
    py = np.array([0.3, 0.4, 0.3])

    # Call the function and print the result
    result = noise_matrix_is_valid(noise_matrix, py, verbose=True)
    print("Is the noise matrix learnable?", result)
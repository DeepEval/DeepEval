import numpy as np

def noise_matrix_is_valid(noise_matrix, py, verbose=False):
    """
    Checks if the given noise_matrix is a learnable matrix.

    Args:
        noise_matrix: A numpy array representing the noise matrix.
        py: A numpy array representing the probabilities of the positive class.
        verbose: A boolean flag indicating whether to print debug messages.

    Returns:
        A boolean value indicating whether the noise matrix is learnable.
    """

    # Calculate the expected probability of the positive class with noise
    expected_py = (1 - noise_matrix) * py + noise_matrix * (1 - py)

    # Check if the expected probability is greater than random performance
    if expected_py.max() <= 0.5:
        if verbose:
            print("Error: Expected probability is not greater than random performance.")
        return False

    return True


if __name__ == "__main__":
    # Create sample input values
    noise_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    py = np.array([0.7, 0.3])

    # Call the function and print the results
    is_learnable = noise_matrix_is_valid(noise_matrix, py)
    print(f"The noise matrix is {'learnable' if is_learnable else 'not learnable'}.")
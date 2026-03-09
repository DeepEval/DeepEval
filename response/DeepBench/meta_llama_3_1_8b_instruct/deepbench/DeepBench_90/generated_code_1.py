import numpy as np

def noise_matrix_is_valid(noise_matrix, py, verbose=False):
    # Check if noise_matrix and py are np.ndarrays
    if not isinstance(noise_matrix, np.ndarray) or not isinstance(py, np.ndarray):
        raise ValueError("Both noise_matrix and py must be np.ndarrays")

    # Get the dimensions of the input matrices
    num_classes = py.shape[0]
    num_samples = py.shape[1]

    # Calculate the probability of each class given the noise matrix and py
    probabilities = np.dot(noise_matrix, py)

    # Calculate the probability of each class given the noise matrix without py
    noise_probabilities = np.dot(noise_matrix, np.ones((num_classes, 1)))

    # Calculate the probability of each class given the noise matrix without py
    noise_probabilities = np.reshape(noise_probabilities, (num_classes,))

    # Check if the probabilities are valid
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1")

    # Check if the probabilities are valid without py
    if np.any(noise_probabilities < 0) or np.any(noise_probabilities > 1):
        raise ValueError("Probabilities without py must be between 0 and 1")

    # Calculate the expected number of samples for each class given the noise matrix and py
    expected_samples = np.dot(py.T, probabilities)

    # Calculate the expected number of samples for each class given the noise matrix without py
    expected_noise_samples = np.dot(np.ones((1, num_samples)), noise_probabilities)

    # Check if the expected number of samples is greater than 0 for all classes
    if np.any(expected_samples <= 0):
        if verbose:
            print("Noise matrix is not learnable because the expected number of samples is not greater than 0 for all classes.")
        return False

    # Check if the expected number of noise samples is greater than 0 for all classes
    if np.any(expected_noise_samples <= 0):
        if verbose:
            print("Noise matrix is not learnable because the expected number of noise samples is not greater than 0 for all classes.")
        return False

    # Check if the expected number of noise samples is greater than the actual number of samples
    if np.any(expected_noise_samples > num_samples):
        if verbose:
            print("Noise matrix is not learnable because the expected number of noise samples is greater than the actual number of samples.")
        return False

    # If all checks pass, the noise matrix is learnable
    return True

if __name__ == "__main__":
    # Create sample input values
    noise_matrix = np.array([[0.1, 0.3], [0.4, 0.5]])
    py = np.array([[0.6, 0.7], [0.2, 0.3]])

    # Call the function and print the results
    print(noise_matrix_is_valid(noise_matrix, py))
    print(noise_matrix_is_valid(noise_matrix, py, verbose=True))
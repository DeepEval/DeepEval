import numpy as np

def noise_matrix_is_valid(noise_matrix, py, verbose=False):
    """
    Check if the given noise_matrix is a learnable matrix.
    
    Parameters:
    - noise_matrix: np.ndarray, a square matrix where noise_matrix[i][j] represents the probability of class i being flipped to class j.
    - py: np.ndarray, a vector of true class probabilities.
    - verbose: bool, optional parameter for printing detailed information.
    
    Returns:
    - bool: True if the noise matrix is learnable, False otherwise.
    """
    
    # Check if noise_matrix is a square matrix
    if noise_matrix.shape[0] != noise_matrix.shape[1]:
        if verbose:
            print("Noise matrix is not square.")
        return False
    
    # Check if the dimensions of noise_matrix and py match
    if noise_matrix.shape[0] != py.shape[0]:
        if verbose:
            print("Dimensions of noise matrix and py do not match.")
        return False
    
    # Calculate the prior distribution of noisy labels
    noisy_label_prior = np.dot(noise_matrix.T, py)
    
    # Calculate the mutual information between true and noisy labels
    mutual_information = np.sum(py * np.log(py / noisy_label_prior))
    
    if verbose:
        print(f"Noise Matrix:\n{noise_matrix}")
        print(f"Prior distribution of noisy labels: {noisy_label_prior}")
        print(f"Mutual information: {mutual_information}")
    
    # The noise matrix is considered learnable if mutual information is positive
    return mutual_information > 0

if __name__ == "__main__":
    # Example noise matrix and class probabilities
    noise_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7]
    ])
    
    py = np.array([0.5, 0.3, 0.2])
    
    # Call the function and print the result
    is_learnable = noise_matrix_is_valid(noise_matrix, py, verbose=True)
    print(f"Is the noise matrix learnable? {is_learnable}")
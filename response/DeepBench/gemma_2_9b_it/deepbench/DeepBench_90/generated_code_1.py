import numpy as np

def noise_matrix_is_valid(noise_matrix, py, verbose=False):
    # Calculate the probability of correct classification under random guessing
    random_guess_prob = 1 / py.shape[1]
    # Calculate the probability of correct classification given the noise matrix
    with np.errstate(divide='ignore', invalid='ignore'):
      conditional_prob = np.sum(noise_matrix * py, axis=1) / np.sum(noise_matrix, axis=1)
    # Identify cases where the conditional probability is significantly better than random guessing
    is_learnable = np.any(conditional_prob > random_guess_prob + 0.1)
    if verbose:
        print(f"Random guess probability: {random_guess_prob}")
        print(f"Conditional probability: {conditional_prob}")
    return is_learnable

if __name__ == "__main__":
    # Sample input values
    noise_matrix = np.array([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]])
    py = np.array([[0.8, 0.1, 0.1],
                  [0.2, 0.7, 0.1]])

    # Call the function and print the result
    is_learnable = noise_matrix_is_valid(noise_matrix, py)
    print(f"Is the noise matrix learnable? {is_learnable}")
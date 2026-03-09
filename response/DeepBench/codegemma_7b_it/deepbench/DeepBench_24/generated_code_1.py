import numpy as np
from sklearn.metrics import confusion_matrix

def compute_confident_joint(latent_true, observed_noisy):
    """
    Estimates the confident counts of latent true vs observed noisy labels.

    Args:
    latent_true: True latent labels (0 or 1).
    observed_noisy: Observed noisy labels (0 or 1).

    Returns:
    A tuple containing the confident counts of latent true vs observed noisy labels.
    """

    # Create a confusion matrix.
    cm = confusion_matrix(latent_true, observed_noisy)

    # Extract the counts of latent true vs observed noisy labels.
    latent_true_positive = cm[1, 1]
    observed_noisy_positive = cm[0, 1]

    return latent_true_positive, observed_noisy_positive

if __name__ == "__main__":
    # Generate sample input values.
    latent_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    observed_noisy = np.array([1, 0, 1, 1, 0, 1, 1, 0])

    # Call the function and print the results.
    latent_true_positive, observed_noisy_positive = compute_confident_joint(latent_true, observed_noisy)
    print("Latent True Positive:", latent_true_positive)
    print("Observed Noisy Positive:", observed_noisy_positive)
import numpy as np

def brier_score_loss(y_true, y_proba, sample_weight=None, pos_label=None):
    """Calculates the Brier score loss.

    Args:
        y_true: True labels, 1D array of shape (n_samples).
        y_proba: Predicted probabilities, 2D array of shape (n_samples, 2).
        sample_weight: Sample weights, optional, 1D array of shape (n_samples).
        pos_label: Positive class label, optional.

    Returns:
        Brier score loss, float.

    Raises:
        TypeError: If y_true or y_proba is not a numpy array.
        ValueError: If y_true contains non-binary classification targets or probability values outside the [0, 1] range.
    """

    # Check input types
    if not isinstance(y_true, np.ndarray) or not isinstance(y_proba, np.ndarray):
        raise TypeError("y_true and y_proba must be numpy arrays.")

    # Check probability values
    if (y_proba < 0).any() or (y_proba > 1).any():
        raise ValueError("Probability values must be between 0 and 1.")

    # Check label consistency
    if pos_label is not None and pos_label not in [0, 1]:
        raise ValueError("pos_label must be either 0 or 1.")
    elif pos_label is None:
        pos_label = 1

    # Calculate Brier score loss
    y_true_encoded = (y_true == pos_label).astype(int)
    loss = np.power(y_true_encoded - y_proba[:, pos_label], 2)

    if sample_weight is not None:
        loss = loss * sample_weight

    return np.mean(loss)


if __name__ == "__main__":
    # Create sample input values
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1], [0.1, 0.9], [0.4, 0.6]])

    # Calculate Brier score loss
    loss = brier_score_loss(y_true, y_proba)

    # Print results
    print("Brier score loss:", loss)
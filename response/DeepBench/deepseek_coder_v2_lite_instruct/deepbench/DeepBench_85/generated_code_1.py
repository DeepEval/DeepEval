import numpy as np

def brier_score_loss(y_true, y_proba, sample_weights=None, pos_label=1):
    # Check input types
    if not isinstance(y_true, (list, np.ndarray)):
        raise TypeError("y_true must be a list or numpy array")
    if not isinstance(y_proba, (list, np.ndarray)):
        raise TypeError("y_proba must be a list or numpy array")
    if sample_weights is not None and not isinstance(sample_weights, (list, np.ndarray)):
        raise TypeError("sample_weights must be a list or numpy array")
    if not isinstance(pos_label, int):
        raise TypeError("pos_label must be an integer")

    # Ensure inputs are numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    if sample_weights is not None:
        sample_weights = np.array(sample_weights)

    # Check if classification is binary
    if not np.all(np.logical_or(y_true == pos_label, y_true == 0)):
        raise ValueError("y_true must contain only binary values or the pos_label")

    # Check if probabilities are within the [0, 1] range
    if not np.all(np.logical_and(y_proba >= 0, y_proba <= 1)):
        raise ValueError("y_proba must be within the [0, 1] range")

    # Check if the number of probabilities matches the number of true labels
    if len(y_true) != len(y_proba):
        raise ValueError("The length of y_true and y_proba must be the same")

    # Calculate the Brier score loss
    squared_errors = (y_true - y_proba) ** 2
    if sample_weights is not None:
        return np.average(squared_errors, weights=sample_weights)
    else:
        return np.mean(squared_errors)

if __name__ == "__main__":
    # Example usage
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.9, 0.3, 0.7]
    print("Brier Score Loss:", brier_score_loss(y_true, y_proba))
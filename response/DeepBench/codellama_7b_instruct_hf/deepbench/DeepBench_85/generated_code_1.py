import numpy as np

def brier_score_loss(y_true, y_proba, sample_weights=None, pos_label=1):
    # Check input types
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array")
    if not isinstance(y_proba, np.ndarray):
        raise ValueError("y_proba must be a numpy array")
    if sample_weights is not None and not isinstance(sample_weights, np.ndarray):
        raise ValueError("sample_weights must be a numpy array")
    if not isinstance(pos_label, int):
        raise ValueError("pos_label must be an integer")

    # Check number of classes
    num_classes = y_true.shape[1]
    if num_classes != 2:
        raise ValueError("y_true must be a binary classification problem")

    # Check probability values
    if not np.all(y_proba >= 0) or not np.all(y_proba <= 1):
        raise ValueError("All probability values must be between 0 and 1")

    # Calculate Brier score loss
    y_true_bin = np.array([1 if label == pos_label else 0 for label in y_true])
    brier_score = np.sum((y_true_bin - y_proba) ** 2)

    # Return Brier score loss
    return brier_score

if __name__ == "__main__":
    # Sample input values
    y_true = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
    y_proba = np.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.4, 0.6]])
    sample_weights = np.array([1, 1, 1, 1])
    pos_label = 1

    # Calculate Brier score loss
    brier_score = brier_score_loss(y_true, y_proba, sample_weights, pos_label)

    # Print Brier score loss
    print(brier_score)
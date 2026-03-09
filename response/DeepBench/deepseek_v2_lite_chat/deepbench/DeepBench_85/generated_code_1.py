import numpy as np
from sklearn.metrics import brier_score_loss

def brier_score_loss(y_true, y_proba, pos_label=None):
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array.")
    if not isinstance(y_proba, np.ndarray):
        raise ValueError("y_proba must be a numpy array.")
    if pos_label is None:
        if np.any(y_proba < 0) or np.any(y_proba > 1):
            raise ValueError("All elements in y_proba must be between 0 and 1.")
    elif pos_label not in set(y_true) or pos_label not in set(y_proba):
        raise ValueError("pos_label must be in the set of actual binary outcomes.")
    return brier_score_loss(y_true, y_proba, pos_label=pos_label)

if __name__ == "__main__":
    # Sample input values
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.4, 0.7])
    pos_label = 0  # Binary classification, where 0 is the negative class

    # Call the function and print the results
    brier_score = brier_score_loss(y_true, y_proba, pos_label)
    print("Brier Score Loss:", brier_score)
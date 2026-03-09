import numpy as np

def brier_score_loss(y_true, y_proba, sample_weight=None, pos_label=1):
    if not isinstance(y_true, (np.ndarray, list)) or not isinstance(y_proba, (np.ndarray, list)):
        raise ValueError("y_true and y_proba must be array-like.")
    
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")

    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("y_true must contain only binary values (0 and 1).")

    if np.any(y_proba < 0) or np.any(y_proba > 1):
        raise ValueError("y_proba values must be in the range [0, 1].")

    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have the same length as y_true.")
        sample_weight = np.array(sample_weight)
    else:
        sample_weight = np.ones_like(y_true)

    score = np.mean(sample_weight * (y_true - y_proba) ** 2) / np.mean(sample_weight)
    return score

if __name__ == "__main__":
    y_true = [0, 1, 1, 0, 1]
    y_proba = [0.1, 0.4, 0.8, 0.2, 0.9]
    score = brier_score_loss(y_true, y_proba)
    print(f"Brier Score Loss: {score}")
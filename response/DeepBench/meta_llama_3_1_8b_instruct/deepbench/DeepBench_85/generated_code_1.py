import numpy as np

def brier_score_loss(y_true, y_proba, sample_weight=None, pos_label=1):
    if not isinstance(y_true, np.ndarray) or not isinstance(y_proba, np.ndarray):
        raise ValueError("Input types are incorrect. Both y_true and y_proba should be numpy arrays.")

    if y_true.dtype!= np.bool_ and y_true.dtype!= np.int_ and y_true.dtype!= np.float_:
        raise ValueError("y_true should be of type bool, int or float.")

    if (y_proba < 0).any() or (y_proba > 1).any():
        raise ValueError("Probability values should be in the range [0, 1].")

    if sample_weight is not None and not isinstance(sample_weight, np.ndarray):
        raise ValueError("sample_weight should be a numpy array if provided.")

    if pos_label not in [0, 1]:
        raise ValueError("pos_label should be either 0 or 1.")

    if pos_label == 0:
        y_true = 1 - y_true
        pos_label = 1

    y_proba = np.array(y_proba)
    y_true = np.array(y_true)

    if sample_weight is not None:
        y_proba = y_proba * sample_weight[:, np.newaxis]
        y_true = y_true * sample_weight

    brier_loss = np.mean((y_true - y_proba) ** 2)
    return brier_loss

if __name__ == "__main__":
    # Create sample input values
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    sample_weight = np.array([1, 1, 2, 2, 1, 1])

    # Call the function and print the results
    brier_loss = brier_score_loss(y_true, y_proba, sample_weight, pos_label=1)
    print("Brier score loss: ", brier_loss)
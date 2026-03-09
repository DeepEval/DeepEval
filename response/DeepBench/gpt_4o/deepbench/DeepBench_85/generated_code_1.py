import numpy as np

def brier_score_loss(y_true, y_proba, sample_weight=None, pos_label=1):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.shape != y_proba.shape:
        raise ValueError("Shapes of y_true and y_proba must be the same.")
    
    if not np.all((y_proba >= 0) & (y_proba <= 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        raise ValueError("Only binary classification is supported.")
    
    if pos_label not in unique_labels:
        raise ValueError("pos_label must be one of the unique values in y_true.")
    
    y_true_binary = np.where(y_true == pos_label, 1, 0)

    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape != y_true.shape:
            raise ValueError("Shapes of sample_weight and y_true must be the same.")
    
    loss = np.average((y_true_binary - y_proba) ** 2, weights=sample_weight)
    return loss

if __name__ == "__main__":
    y_true = [0, 1, 1, 0]
    y_proba = [0.1, 0.9, 0.8, 0.3]
    sample_weight = [0.5, 1, 1, 0.5]
    pos_label = 1
    
    score = brier_score_loss(y_true, y_proba, sample_weight=sample_weight, pos_label=pos_label)
    print(f"Brier score loss: {score}")
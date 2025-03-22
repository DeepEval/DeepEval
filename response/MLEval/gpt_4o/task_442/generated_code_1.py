import numpy as np

def my_iou_metric_2(preds, targets, threshold=0.5):
    """
    Example IoU metric calculation function.
    Assumes binary predictions and targets.
    """
    preds = (preds > threshold).astype(np.int)
    intersection = np.sum(preds & targets)
    union = np.sum(preds | targets)
    iou = intersection / union if union != 0 else 0
    return iou

def method(preds, targets):
    """
    Method to calculate IoU with a default threshold of 0.
    """
    # Set the default threshold to 0 instead of 0.5
    threshold = 0.0
    # Use the modified threshold in our IoU metric function
    iou = my_iou_metric_2(preds, targets, threshold=threshold)
    return iou

# Sample data for validation
preds = np.array([[0.1, 0.7, 0.3], [0.6, 0.2, 0.8], [0.4, 0.9, 0.5]])
targets = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

# Validate the method
output = method(preds, targets)
print("IoU with default threshold 0:", output)
import numpy as np

def calculate_iou(y_true, y_pred, threshold=0):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.
    
    Parameters:
    y_true (numpy array): Ground truth binary mask.
    y_pred (numpy array): Predicted binary mask.
    threshold (float): Threshold value for prediction (default is 0).
    
    Returns:
    float: IoU score.
    """
    # Apply threshold to predictions
    y_pred_thresholded = (y_pred > threshold).astype(int)
    
    # Calculate intersection and union
    intersection = np.logical_and(y_true, y_pred_thresholded)
    union = np.logical_or(y_true, y_pred_thresholded)
    
    # Calculate IoU
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)  # Add epsilon to avoid division by zero
    return iou

def method():
    # Example usage:
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    y_pred = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    
    iou = calculate_iou(y_true, y_pred, threshold=0)
    output = iou
    return output

# Call the method for validation
result = method()
print("IoU:", result)
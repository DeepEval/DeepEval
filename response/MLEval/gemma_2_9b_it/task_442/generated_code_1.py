import numpy as np

def method():
  """
  Calculates the Intersection over Union (IoU) metric for pixel-level predictions.

  This function uses a threshold of 0 for pixel prediction,
  different from the default 0.5 used in my_iou_metric_2.
  """
  # Example usage: Assuming you have ground truth and predicted pixel maps
  ground_truth = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
  ])
  predicted = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
  ])

  # Apply thresholding for predictions
  predicted_thresholded = (predicted > 0).astype(int)

  # Calculate Intersection and Union
  intersection = np.logical_and(ground_truth, predicted_thresholded).sum()
  union = np.logical_or(ground_truth, predicted_thresholded).sum()

  # Calculate IoU
  iou = intersection / union if union > 0 else 0  

  return iou

# Validate the method
output = method()
print(f"IoU using threshold 0: {output}")
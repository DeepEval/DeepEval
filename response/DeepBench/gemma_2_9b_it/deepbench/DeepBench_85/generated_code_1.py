def brier_score_loss(y_true, y_proba, sample_weight=None, pos_label=1):
  import numpy as np
  if not isinstance(y_true, np.ndarray):
    raise TypeError("y_true must be a NumPy array.")
  if y_true.ndim != 1:
    raise ValueError("y_true must be one-dimensional.")
  if not isinstance(y_proba, np.ndarray):
    raise TypeError("y_proba must be a NumPy array.")
  if y_proba.ndim != 1:
    raise ValueError("y_proba must be one-dimensional.")
  if len(y_true) != len(y_proba):
    raise ValueError("Length of y_true and y_proba must be equal.")
  if np.any(y_proba < 0) or np.any(y_proba > 1):
    raise ValueError("y_proba must be between 0 and 1.")
  if sample_weight is not None:
    if not isinstance(sample_weight, np.ndarray):
      raise TypeError("sample_weight must be a NumPy array.")
    if len(sample_weight) != len(y_true):
      raise ValueError("Length of sample_weight must be equal to y_true.")
  
  return np.mean((y_true - y_proba)**2 * (sample_weight if sample_weight is not None else np.ones_like(y_true)))

if __name__ == "__main__":
  y_true = np.array([0, 1, 1, 0])
  y_proba = np.array([0.2, 0.8, 0.9, 0.1])
  loss = brier_score_loss(y_true, y_proba)
  print(f"Brier Score Loss: {loss}")
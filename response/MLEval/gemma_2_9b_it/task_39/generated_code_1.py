import numpy as np

def method(y_true, y_pred):
  """
  Calculates the Coefficient of Determination (R-squared).

  Args:
      y_true: A NumPy array of the true target values.
      y_pred: A NumPy array of the predicted target values.

  Returns:
      float: The R-squared value.
  """
  y_mean = np.mean(y_true)
  ssr = np.sum((y_pred - y_mean)**2)
  sst = np.sum((y_true - y_mean)**2)
  r_squared = 1 - (ssr / sst)
  return r_squared

# Example usage:
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.2, 1.9, 3.1, 3.8, 4.9])

r2 = method(y_true, y_pred)
print(f"R-squared: {r2}")
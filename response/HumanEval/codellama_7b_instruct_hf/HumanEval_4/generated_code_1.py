import numpy as np

def method(input_numbers):
  """Calculates the Mean Absolute Deviation (MAD) around the mean of a dataset.

  Args:
    input_numbers: A list of numbers.

  Returns:
    The Mean Absolute Deviation of the dataset.
  """

  # Calculate the mean of the dataset
  mean = np.mean(input_numbers)

  # Calculate the absolute difference between each element and the mean
  absolute_differences = np.abs(input_numbers - mean)

  # Calculate the mean of the absolute differences
  mad = np.mean(absolute_differences)

  return mad

# Test case
input_numbers = [1, 2, 3, 4, 5]
expected_mad = 1.6
actual_mad = method(input_numbers)

if expected_mad == actual_mad:
  print("Test case passed.")
else:
  print(f"Test case failed. Expected {expected_mad}, but got {actual_mad}.")
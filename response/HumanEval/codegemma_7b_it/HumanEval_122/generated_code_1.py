import numpy as np

def method(arr, k):
  output = 0
  for i in range(k):
    if len(str(arr[i])) <= 2:
      output += arr[i]
  return output


# Test case
arr = np.array([123, 45, 6789, 10, 12, 13, 14, 15])
k = 5

print(method(arr, k))
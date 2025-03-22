import numpy as np

def method():
  X = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

  # Compute the sum along the second and fourth elements
  sum1 = np.sum(X[:, [1, 3]], axis=1) 

  # Compute the sum along the first and third elements
  sum2 = np.sum(X[:, [0, 2]], axis=1)

  output = np.concatenate((sum1, sum2))
  return output

# Call the method and print the output
output = method()
print(output)
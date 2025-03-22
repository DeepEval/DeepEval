import numpy as np

def method():
  
  # Replace this with your actual matrix X
  X = np.array([[1, 2], [3, 4]]) 
  
  eigenvalues, eigenvectors = np.linalg.eig(X)
  
  output = {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}
  return output

# Call the method and print the result
result = method()
print(result)
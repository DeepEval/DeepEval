import numpy as np

# Define the function method()
def method():
    # Define the input array X
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    
    # Compute the sum along the second and fourth elements of X
    sum_second_fourth = np.sum(X[:, [1, 3]], axis=1)
    
    # Compute the sum along the first and third elements of X
    sum_first_third = np.sum(X[:, [0, 2]], axis=1)
    
    # Stack the results vertically
    output = np.vstack((sum_second_fourth, sum_first_third))
    
    return output

# Call the method() function for validation
output = method()
print(output)
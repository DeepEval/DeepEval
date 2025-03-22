import numpy as np

def method():
    # Create a sample 2D array X
    X = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

    # Compute the sum along the second and fourth elements of X
    sum_second_fourth = np.sum(X[:, [1, 3]], axis=1)

    # Compute the sum along the first and third elements of X
    sum_first_third = np.sum(X[:, [0, 2]], axis=1)

    # Combine the sums in the desired order
    output = np.concatenate((sum_first_third, sum_second_fourth))

    # Return the output
    return output

# Call the method for validation
output = method()
print(output)
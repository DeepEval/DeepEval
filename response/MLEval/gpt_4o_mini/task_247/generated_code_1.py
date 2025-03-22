import numpy as np

def method():
    # Example input array X
    # Let's create a 1D array with at least 4 elements for demonstration
    X = np.array([1, 2, 3, 4, 5, 6])

    # Compute the sum of the second (index 1) and fourth (index 3) elements
    sum_2_and_4 = X[1] + X[3]

    # Compute the sum of the first (index 0) and third (index 2) elements
    sum_1_and_3 = X[0] + X[2]

    # Prepare the output
    output = {
        'sum_2_and_4': sum_2_and_4,
        'sum_1_and_3': sum_1_and_3
    }

    return output

# Call the method for validation
result = method()
print(result)
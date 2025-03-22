def method():
    import numpy as np

    # Example input array
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Compute the sum along the second and fourth elements of each row
    sum_second_fourth = X[:, 1] + X[:, 3]

    # Compute the sum along the first and third elements of each row
    sum_first_third = X[:, 0] + X[:, 2]

    # Combine the results into a single output
    output = {
        "sum_second_fourth": sum_second_fourth,
        "sum_first_third": sum_first_third
    }

    return output

# Call the method for validation
result = method()
print(result)
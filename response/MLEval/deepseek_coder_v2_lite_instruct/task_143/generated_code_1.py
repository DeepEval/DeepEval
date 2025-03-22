def method():
    import numpy as np

    # Example input arrays
    x = np.array([10, 20, 30])
    y = np.array([3, 5, 7])

    # Compute the remainder element-wise
    output = np.mod(x, y)

    return output

# Call the method for validation
result = method()
print(result)  # Output should be [1, 0, 2]
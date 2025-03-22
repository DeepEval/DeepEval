import numpy as np

def method():
    # Example tensor X with shape (..., a, b)
    X = np.array([
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [7, 8]
        ]
    ])

    # Transpose the last two dimensions
    output = X.transpose(-1, -2)

    return output

# Call the method and print the result
output = method()
print(output)
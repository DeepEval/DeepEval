import numpy as np

def pyramid_combination(values, weight_floor, weight_ceil):
    # Check input shapes are compatible
    if len(values) != 2**len(weight_floor):
        raise ValueError("Number of values must match the number of weights")
    if len(values) != 2**len(weight_ceil):
        raise ValueError("Number of values must match the number of weights")
    if len(weight_floor) != len(weight_ceil):
        raise ValueError("Number of weights must be the same for both floors and ceils")

    # Calculate the interpolation weights
    weight_floor = np.array(weight_floor)
    weight_ceil = np.array(weight_ceil)
    weights = weight_floor + weight_ceil
    weights /= np.sum(weights, axis=1, keepdims=True)

    # Combine the values
    output = np.zeros((len(values), len(values[0])))
    for i in range(len(values)):
        output[i] = np.sum(values[i] * weights[i], axis=0)

    return output

if __name__ == "__main__":
    # Generate some sample input values
    values = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]]),
        np.array([[13, 14], [15, 16]]),
    ]

    # Define the interpolation weights
    weight_floor = [
        np.array([[0.25, 0.75], [0.75, 0.25]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.75, 0.25], [0.25, 0.75]]),
        np.array([[0.8, 0.2], [0.2, 0.8]]),
    ]
    weight_ceil = [
        np.array([[0.25, 0.75], [0.75, 0.25]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.75, 0.25], [0.25, 0.75]]),
        np.array([[0.8, 0.2], [0.2, 0.8]]),
    ]

    # Call the function
    output = pyramid_combination(values, weight_floor, weight_ceil)

    # Print the output
    print(output)
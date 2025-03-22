import numpy as np

def method(input_numbers):
    """
    Calculate Mean Absolute Deviation around the mean of the given dataset.

    Parameters:
    input_numbers (list): A list of input numbers.

    Returns:
    float: The Mean Absolute Deviation of the input numbers.
    """
    # Check if the input list is empty
    if not input_numbers:
        return 0

    # Calculate the mean of the input numbers
    mean = np.mean(input_numbers)

    # Calculate the absolute differences between each element and the mean
    absolute_diffs = [abs(x - mean) for x in input_numbers]

    # Calculate the Mean Absolute Deviation
    mad = np.mean(absolute_diffs)

    return mad


# Test case for validation
input_numbers = [1, 2, 3, 4, 5]
output = method(input_numbers)
print(f"Input: {input_numbers}")
print(f"Mean Absolute Deviation: {output}")
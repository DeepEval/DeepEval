def method():
    """
    Compute the average of integers from n through m, round to the nearest integer, 
    convert to binary and return. If n is greater than m, return -1.

    Args:
    n (int): The start of the range (inclusive)
    m (int): The end of the range (inclusive)

    Returns:
    str: The binary representation of the rounded average
    """

    # Import necessary packages
    import math

    # Define the function parameters
    n = 5  # Start of the range
    m = 10  # End of the range

    # Check if n is greater than m
    if n > m:
        return -1

    # Calculate the sum of the range
    total = sum(range(n, m + 1))

    # Calculate the average
    average = total / (m - n + 1)

    # Round the average to the nearest integer
    rounded_average = math.floor(average + 0.5)

    # Convert the rounded average to binary
    binary_average = bin(rounded_average)[2:]

    # Return the binary representation of the rounded average
    return binary_average

# Test case
output = method()
print("Output:", output)
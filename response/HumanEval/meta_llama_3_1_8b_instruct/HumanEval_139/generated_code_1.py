import math

def method(n):
    """
    Calculate the Brazilian factorial of a given number n.

    Args:
    n (int): The number for which to calculate the Brazilian factorial.

    Returns:
    int: The Brazilian factorial of n.

    Raises:
    ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")

    # Initialize the result with 1
    result = 1
    
    # Calculate the factorial from n down to 1 and multiply it with the result
    for i in range(n, 0, -1):
        # Calculate the factorial of i using math.factorial
        result *= math.factorial(i)
    
    return result

# Test case
try:
    n = 5
    output = method(n)
    print(f"The Brazilian factorial of {n} is: {output}")
except ValueError as e:
    print(f"Error: {e}")
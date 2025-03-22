def method(n):
    """
    Returns the n-th Fibonacci number.

    Args:
        n (int): The position of the Fibonacci number to return.

    Returns:
        int: The n-th Fibonacci number.

    Raises:
        ValueError: If n is a negative integer.
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    
    # Base cases
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    # Initialize variables
    a, b = 0, 1
    
    # Calculate the n-th Fibonacci number
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

# Test case
print(method(10))  # Output: 55
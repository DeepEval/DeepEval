def sum_to_n(n):
    """
    This function calculates the sum of numbers from 1 to n.
    
    Args:
        n (int): The upper limit of the range.
    
    Returns:
        int: The sum of numbers from 1 to n.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("Input should be a positive integer.")
    
    total = 0
    for i in range(1, n + 1):
        total += i
    
    return total

# Test case
n = 10
output = sum_to_n(n)
print(f"The sum of numbers from 1 to {n} is: {output}")
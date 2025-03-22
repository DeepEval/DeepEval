def method(n):
    """
    This function finds the largest number that divides n evenly, smaller than n.

    Args:
        n (int): The input number.

    Returns:
        int: The largest number that divides n evenly, smaller than n.
    """
    # Initialize the largest divisor as 1
    largest_divisor = 1

    # Iterate from n-1 to 2 in reverse order
    for i in range(n-1, 1, -1):
        # Check if i is a divisor of n
        if n % i == 0:
            # Update the largest divisor
            largest_divisor = i
            # Break the loop as we found the largest divisor
            break

    return largest_divisor


# Test case
n = 20
output = method(n)
print(f"The largest number that divides {n} evenly, smaller than {n} is: {output}")
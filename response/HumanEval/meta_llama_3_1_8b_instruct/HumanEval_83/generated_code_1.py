def method(n):
    """
    This function calculates the count of the numbers of n-digit positive integers 
    that start or end with 1.
    
    Parameters:
    n (int): The number of digits in the positive integers.
    
    Returns:
    int: The count of the numbers of n-digit positive integers that start or end with 1.
    """
    
    # Import necessary modules
    import math
    
    # Calculate the total number of n-digit positive integers
    total_count = 9 * (10 ** (n - 2)) + 1
    
    # Calculate the count of n-digit positive integers that start with 1
    start_with_1_count = 1 * (10 ** (n - 1))
    
    # Calculate the count of n-digit positive integers that end with 1
    end_with_1_count = 9 * (10 ** (n - 2))
    
    # The count of n-digit positive integers that start or end with 1 is the sum of 
    # the count of n-digit positive integers that start with 1 and end with 1, 
    # minus the count of n-digit positive integers that start and end with 1
    overlap_count = 9 * (10 ** (n - 3))
    
    return start_with_1_count + end_with_1_count - overlap_count


# Test case
n = 3
output = method(n)
print("The count of the numbers of", n, "-digit positive integers that start or end with 1 is:", output)
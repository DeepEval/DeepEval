def method():
    """
    This function calculates the n-th element of the fib4 number sequence.
    
    The fib4 number sequence is a sequence similar to the Fibonacci sequence that's defined as follows:
    fib4(0) -> 0 
    fib4(1) -> 0 
    fib4(2) -> 2 
    fib4(3) -> 0 
    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).
    
    This function uses dynamic programming to efficiently compute the n-th element of the fib4 number sequence.
    
    Parameters:
    n (int): The position of the element in the sequence.
    
    Returns:
    int: The n-th element of the fib4 number sequence.
    """
    
    # Initialize the base cases
    fib4 = [0, 0, 2]
    
    # If n is less than or equal to 3, return the n-th element directly
    if n <= 3:
        return fib4[n]
    
    # Initialize the variables to store the previous four elements
    a, b, c, d = fib4
    
    # Compute the n-th element
    for i in range(3, n):
        # Calculate the next element as the sum of the previous four elements
        a, b, c, d = b, c, d, a + b + c + d
    
    # Return the n-th element
    return d


# Test case
print(method(5))  # Output: 4
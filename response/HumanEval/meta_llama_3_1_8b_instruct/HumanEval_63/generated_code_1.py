def method():
    """
    This function efficiently computes the n-th element of the FibFib number sequence.
    
    The FibFib number sequence is a sequence similar to the Fibonacci sequence that's defined as follows:
    fibfib(0) == 0 
    fibfib(1) == 0 
    fibfib(2) == 1 
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3)
    
    Parameters:
    n (int): The position of the element in the sequence.
    
    Returns:
    int: The n-th element of the FibFib number sequence.
    """
    
    # Define a dictionary to store the computed values to avoid repeated computations
    memo = {0: 0, 1: 0, 2: 1}
    
    def fibfib(n):
        # If the value is already computed, return it from the dictionary
        if n in memo:
            return memo[n]
        
        # If the value is not computed, compute it and store it in the dictionary
        result = fibfib(n-1) + fibfib(n-2) + fibfib(n-3)
        memo[n] = result
        
        return result
    
    # Return the n-th element of the FibFib number sequence
    return fibfib(n)

# Test case for validation
n = 10
output = method()
print(f"The {n}-th element of the FibFib number sequence is: {output}")
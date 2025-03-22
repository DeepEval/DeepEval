def method():
    # Initialize a dictionary to store computed values
    memo = {0: 0, 1: 0, 2: 1}
    
    def fibfib(n):
        if n in memo:
            return memo[n]
        memo[n] = fibfib(n-1) + fibfib(n-2) + fibfib(n-3)
        return memo[n]
    
    # Compute the 5th element of the sequence
    output = fibfib(5)
    return output

# Test case
print(method())  # Expected output: 4
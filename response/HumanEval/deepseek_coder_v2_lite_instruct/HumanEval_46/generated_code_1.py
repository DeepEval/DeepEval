def method():
    # Define the sequence starting values
    fib4_sequence = [0, 0, 2, 0]
    
    # Get the value of n from the user (for simplicity, we'll use a hardcoded value here)
    n = 10  # Example value
    
    # Compute the sequence up to the n-th element
    for i in range(4, n + 1):
        next_value = fib4_sequence[i - 1] + fib4_sequence[i - 2] + fib4_sequence[i - 3] + fib4_sequence[i - 4]
        fib4_sequence.append(next_value)
    
    # Return the n-th element of the sequence
    output = fib4_sequence[n]
    return output

# Test the function with a known value
print(method())  # Should print the 10th element of the Fib4 sequence
def method(n):
    # Handle base cases
    if n == 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 2
    elif n == 3:
        return 0
    
    # Create an array to store the computed values
    fib4_values = [0] * (n + 1)
    
    # Initialize the base cases
    fib4_values[0] = 0
    fib4_values[1] = 0
    fib4_values[2] = 2
    fib4_values[3] = 0
    
    # Compute the fib4 values iteratively
    for i in range(4, n + 1):
        fib4_values[i] = fib4_values[i - 1] + fib4_values[i - 2] + fib4_values[i - 3] + fib4_values[i - 4]
    
    return fib4_values[n]

# Test case for validation
if __name__ == "__main__":
    n = 10  # Example input
    output = method(n)
    print(f"The {n}-th element of the Fib4 sequence is: {output}")
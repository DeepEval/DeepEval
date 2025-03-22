def method(n):
    # Handle base cases
    if n == 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    
    # Initialize the first three elements of the sequence
    fibfib_values = [0, 0, 1]
    
    # Calculate fibfib values up to the n-th element
    for i in range(3, n + 1):
        next_value = fibfib_values[i - 1] + fibfib_values[i - 2] + fibfib_values[i - 3]
        fibfib_values.append(next_value)
    
    # The n-th element is the last element in the list
    return fibfib_values[n]

# Test case for validation
n = 10
output = method(n)
print(f"The {n}-th element of the FibFib sequence is: {output}")
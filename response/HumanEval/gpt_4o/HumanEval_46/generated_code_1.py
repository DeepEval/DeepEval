def method(n):
    # Base cases
    if n == 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 2
    elif n == 3:
        return 0

    # Initialize the first four numbers of the Fib4 sequence
    fib4_sequence = [0, 0, 2, 0]

    # Compute the Fib4 sequence up to the n-th element iteratively
    for i in range(4, n + 1):
        next_value = fib4_sequence[i - 1] + fib4_sequence[i - 2] + fib4_sequence[i - 3] + fib4_sequence[i - 4]
        fib4_sequence.append(next_value)

    # Return the n-th element of the sequence
    return fib4_sequence[n]

# Test case to validate the function
test_case = 10
output = method(test_case)
print(f"Fib4({test_case}) = {output}")
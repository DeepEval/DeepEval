def method(n):
    # Initialize an array to store the Fib4 sequence
    fib4 = [0] * (n + 1)

    # Base cases
    fib4[2] = 2
    fib4[3] = 0

    # Fill the array using the recurrence relation
    for i in range(4, n + 1):
        fib4[i] = fib4[i - 1] + fib4[i - 2] + fib4[i - 3] + fib4[i - 4]

    # The n-th element is the last element in the array
    output = fib4[n]
    return output

# Test case
print(method(10))  # Output should be the 10-th element of the Fib4 sequence
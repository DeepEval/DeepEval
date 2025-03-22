import numpy as np

def fib4(n):
    if n == 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 2
    elif n == 3:
        return 0
    else:
        # Initialize the first four elements
        fib4_sequence = [0, 0, 2, 0]

        # Calculate the remaining elements using the formula
        for i in range(4, n + 1):
            fib4_sequence.append(fib4_sequence[i - 1] + fib4_sequence[i - 2] + fib4_sequence[i - 3] + fib4_sequence[i - 4])

        # Return the nth element in the sequence
        return fib4_sequence[n]

# Test case
n = 5
output = fib4(n)

print(output)
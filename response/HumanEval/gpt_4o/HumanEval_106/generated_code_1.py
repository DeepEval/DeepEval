import math

def method(n):
    output = []
    for i in range(1, n + 1):
        if i % 2 == 0:
            # Calculate factorial of i
            value = math.factorial(i)
        else:
            # Calculate the sum from 1 to i
            value = sum(range(1, i + 1))
        output.append(value)
    return output

# Example test case for validation
n = 5
result = method(n)
print(result)  # Expected output: [1, 2, 6, 24, 15]
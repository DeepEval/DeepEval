import math

def f(n):
    result = []
    for i in range(1, n + 1):
        if i % 2 == 0:  # i is even
            result.append(math.factorial(i))
        else:  # i is odd
            result.append(sum(range(1, i + 1)))
    return result

def method():
    # Example test case
    n = 5
    output = f(n)
    return output

# Running the test case
print(method())
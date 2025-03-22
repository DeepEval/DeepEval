import math

def method(x, y):

    # Check if both x and y are positive integers
    if not (isinstance(x, int) and isinstance(y, int) and x > 0 and y > 0):
        return -1

    # Check if x is greater than y
    if x > y:
        x, y = y, x

    # Find the next even integer greater than or equal to x
    even_x = math.ceil(x / 2) * 2

    # Iterate through the range [x, y] in steps of 2
    for i in range(even_x, y + 1, 2):
        if i % 2 == 0:
            output = i

    # If no even integer is found, return -1
    if output is None:
        output = -1

    return output

# Test case
x = 10
y = 20

print(method(x, y))  # Output: 20
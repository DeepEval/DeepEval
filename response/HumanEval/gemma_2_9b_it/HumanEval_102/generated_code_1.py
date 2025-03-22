def method(x, y):
    """
    This function takes two positive numbers x and y and returns the biggest even integer number that is in the range [x, y] inclusive. 
    If there's no such number, then the function should return -1. 
    """
    for i in range(y, x - 1, -1):
        if i % 2 == 0:
            return i
    return -1

# Test Case
x = 5
y = 10
output = method(x, y)
print(f"The biggest even number between {x} and {y} is: {output}")
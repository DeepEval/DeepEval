def method(x, y):
    # Ensure x is less than or equal to y
    if x > y:
        return -1  # or raise an Exception if preferred

    # Find the largest even integer in the range
    for number in range(y, x - 1, -1):
        if number % 2 == 0:
            return number
    
    return -1  # If no even number found

# Test case
x = 5
y = 15
output = method(x, y)
print("The largest even number between", x, "and", y, "is:", output)
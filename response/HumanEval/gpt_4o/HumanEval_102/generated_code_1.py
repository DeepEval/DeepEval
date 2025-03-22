def method(x, y):
    # Ensure x and y are positive and x <= y
    if x > y or x <= 0 or y <= 0:
        return -1
    
    # Start from y and go down to x
    for num in range(y, x - 1, -1):
        if num % 2 == 0:
            return num
    
    # If no even number is found
    return -1

# Test case for validation
# For x = 3 and y = 10, the biggest even integer is 10
print(method(3, 10))  # Expected output: 10

# Additional test case
# For x = 5 and y = 9, the biggest even integer is 8
print(method(5, 9))  # Expected output: 8

# When x = 7 and y = 7, there are no even numbers
print(method(7, 7))  # Expected output: -1

# When x = 1 and y = 2, the biggest even integer is 2
print(method(1, 2))  # Expected output: 2
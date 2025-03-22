def method(a, b, c):
    # Check if all inputs are integers
    if not all(isinstance(n, int) for n in [a, b, c]):
        return False
    
    # Check if one number is the sum of the other two
    if a == b + c or b == a + c or c == a + b:
        return True
    else:
        return False

# Test case
output = method(3, 5, 2)  # True, because 5 is equal to 3 + 2
print(output)  # Expected output: True
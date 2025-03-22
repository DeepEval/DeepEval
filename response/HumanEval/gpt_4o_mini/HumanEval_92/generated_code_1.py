def method(a, b, c):
    # Check if all inputs are integers
    if isinstance(a, int) and isinstance(b, int) and isinstance(c, int):
        # Check if one number is equal to the sum of the other two
        return a == b + c or b == a + c or c == a + b
    else:
        return False

# Test case validation
output = method(3, 5, 8)  # Should return True (3 + 5 = 8)
print(output)  # Output: True

output = method(1, 2, 4)  # Should return False (1 + 2 != 4)
print(output)  # Output: False

output = method(10, 5, 5)  # Should return True (5 + 5 = 10)
print(output)  # Output: True

output = method(1.5, 2, 3)  # Should return False (1.5 is not an integer)
print(output)  # Output: False
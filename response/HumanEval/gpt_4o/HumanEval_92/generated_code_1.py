def method(num1, num2, num3):
    # Check if all numbers are integers
    if isinstance(num1, int) and isinstance(num2, int) and isinstance(num3, int):
        # Check if one of the numbers is the sum of the other two
        if num1 == num2 + num3 or num2 == num1 + num3 or num3 == num1 + num2:
            return True
    return False

# Test case for validation
if __name__ == "__main__":
    # Test cases
    print(method(5, 2, 3))  # Expected output: True, because 5 = 2 + 3
    print(method(10, 5, 5))  # Expected output: True, because 10 = 5 + 5
    print(method(7, 3, 4))  # Expected output: True, because 7 = 3 + 4
    print(method(4, 2, 1))  # Expected output: False, no number is the sum of the other two
    print(method(6, 1, 5))  # Expected output: True, because 6 = 1 + 5
    print(method(2.5, 1, 1.5))  # Expected output: False, because not all numbers are integers
    print(method(0, -2, 2))  # Expected output: True, because 0 = -2 + 2
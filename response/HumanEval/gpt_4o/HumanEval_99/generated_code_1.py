def method(value):
    # Convert the string to a floating-point number
    number = float(value)

    # Use the built-in round function to get the nearest integer
    rounded = round(number)

    # Check if the number is exactly halfway between two integers
    if abs(number - int(number)) == 0.5:
        # If it is, round away from zero
        if number > 0:
            rounded = int(number) + 1
        else:
            rounded = int(number) - 1

    return rounded

# Test case for validation
def test_method():
    # Test cases to validate the function
    test_cases = [
        ("2.5", 3),
        ("-2.5", -3),
        ("4.4", 4),
        ("-4.4", -4),
        ("5.6", 6),
        ("-5.6", -6),
        ("3.0", 3),
        ("-3.0", -3),
    ]

    for value, expected in test_cases:
        result = method(value)
        assert result == expected, f"Test failed for value: {value}. Expected: {expected}, got: {result}"

    print("All test cases passed!")

# Run the test
test_method()
def method(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Test case for validation
def test_method():
    # Test case 1
    a = 48
    b = 18
    expected_output = 6
    assert method(a, b) == expected_output, f"Test case 1 failed: expected {expected_output}, got {method(a, b)}"

    # Test case 2
    a = 100
    b = 25
    expected_output = 25
    assert method(a, b) == expected_output, f"Test case 2 failed: expected {expected_output}, got {method(a, b)}"

    # Test case 3
    a = 7
    b = 5
    expected_output = 1
    assert method(a, b) == expected_output, f"Test case 3 failed: expected {expected_output}, got {method(a, b)}"

    # Test case 4
    a = 0
    b = 10
    expected_output = 10
    assert method(a, b) == expected_output, f"Test case 4 failed: expected {expected_output}, got {method(a, b)}"

    # Test case 5
    a = 17
    b = 0
    expected_output = 17
    assert method(a, b) == expected_output, f"Test case 5 failed: expected {expected_output}, got {method(a, b)}"

    print("All test cases passed.")

# Run the test cases
test_method()
def method(n):
    # Generate numbers from 0 to n
    numbers = range(n + 1)
    # Convert numbers to a space-delimited string
    output = ' '.join(map(str, numbers))
    return output

# Test case for validation
def test_method():
    # Test with n = 5
    n = 5
    expected_output = "0 1 2 3 4 5"
    assert method(n) == expected_output, f"Test failed: expected {expected_output}, got {method(n)}"
    
    # Test with n = 0
    n = 0
    expected_output = "0"
    assert method(n) == expected_output, f"Test failed: expected {expected_output}, got {method(n)}"
    
    # Test with n = 10
    n = 10
    expected_output = "0 1 2 3 4 5 6 7 8 9 10"
    assert method(n) == expected_output, f"Test failed: expected {expected_output}, got {method(n)}"

    print("All tests passed.")

# Run the test case
test_method()
def method(xs, x):
    # Initialize the output as 0
    output = 0
    
    # Iterate over the coefficients and their indices
    for i, coeff in enumerate(xs):
        # Add the contribution of each term to the output
        output += coeff * (x ** i)
    
    return output

# Test case
def test_method():
    # Coefficients for the polynomial: 3 + 2x + x^2
    coefficients = [3, 2, 1]
    point = 2
    # The polynomial value at x = 2 should be 3 + 2*2 + 1*2^2 = 3 + 4 + 4 = 11
    expected_output = 11
    
    # Call the method
    result = method(coefficients, point)
    
    # Verify the result
    assert result == expected_output, f"Test failed: expected {expected_output}, got {result}"
    print("Test passed.")

# Run the test
test_method()
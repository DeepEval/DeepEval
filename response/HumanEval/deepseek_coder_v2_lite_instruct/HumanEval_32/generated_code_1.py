def method():
    # Example coefficients and point
    xs = [1, 2, 3, 4]  # Coefficients of the polynomial
    x = 2              # Point at which to evaluate the polynomial

    # Initialize the result
    result = 0

    # Evaluate the polynomial
    for i, coeff in enumerate(xs):
        result += coeff * (x ** i)

    # Return the result
    return result

# # Test case
# def test_method():
#     assert method() == 1 + 2*2 + 3*2**2 + 4*2**3
#     print("Test passed!")

# # Run the test case
# test_method()

# Run the method
method()
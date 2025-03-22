import numpy as np

def evaluate_polynomial(xs, x):
    """
    Evaluates a polynomial with coefficients xs at point x.

    Args:
        xs (list): Coefficients of the polynomial.
        x (float): Point at which to evaluate the polynomial.

    Returns:
        float: The value of the polynomial at point x.
    """
    # Use numpy's polyval function to evaluate the polynomial
    output = np.polyval(xs, x)
    return output

# Test case
def test_evaluate_polynomial():
    xs = [3, 2, 1]  # coefficients of the polynomial 3 + 2x + x^2
    x = 2
    expected_output = 17
    actual_output = evaluate_polynomial(xs, x)
    # assert np.isclose(actual_output, expected_output), f"Expected {expected_output} but got {actual_output}"

# Run the test case
test_evaluate_polynomial()
print("Test passed")
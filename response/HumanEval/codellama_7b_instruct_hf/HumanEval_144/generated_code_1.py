from fractions import Fraction

def simplify(x, n):
    """
    Simplifies the expression x * n.

    Args:
        x: The first fraction as a string.
        n: The second fraction as a string.

    Returns:
        True if x * n evaluates to a whole number and False otherwise.
    """

    # Create Fraction objects from the string inputs.
    x_frac = Fraction(x)
    n_frac = Fraction(n)

    # Multiply the fractions together.
    result = x_frac * n_frac

    # Check if the result is a whole number.
    if result.numerator % result.denominator == 0:
        return True
    else:
        return False


# Test case
x = "1/5"
n = "5/1"
result = simplify(x, n)

print(result)  # Output: True
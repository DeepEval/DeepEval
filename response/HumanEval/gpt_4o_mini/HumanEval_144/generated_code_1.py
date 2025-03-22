def method(x, n):
    # Split the fractions into numerator and denominator
    x_numerator, x_denominator = map(int, x.split('/'))
    n_numerator, n_denominator = map(int, n.split('/'))

    # Calculate the product of the fractions
    # (x_numerator / x_denominator) * (n_numerator / n_denominator) = (x_numerator * n_numerator) / (x_denominator * n_denominator)
    product_numerator = x_numerator * n_numerator
    product_denominator = x_denominator * n_denominator

    # Check if product_numerator is divisible by product_denominator
    return product_numerator % product_denominator == 0

# Test cases for validation
print(method("1/5", "5/1"))  # True
print(method("1/6", "2/1"))  # False
print(method("7/10", "10/2"))  # False
print(method("3/4", "8/3"))  # True (3/4 * 8/3 = 24/12 = 2, a whole number)
print(method("2/5", "5/2"))  # True (2/5 * 5/2 = 10/10 = 1, a whole number)

# Output the results
output = [method("1/5", "5/1"), method("1/6", "2/1"), method("7/10", "10/2"),
          method("3/4", "8/3"), method("2/5", "5/2")]
print(output)
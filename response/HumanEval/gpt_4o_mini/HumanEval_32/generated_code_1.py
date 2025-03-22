def method(xs, x):
    # Evaluate the polynomial at point x using the coefficients xs
    output = 0
    for i in range(len(xs)):
        output += xs[i] * (x ** i)
    return output

# Test case
coefficients = [2, 3, 5]  # Represents the polynomial 2 + 3*x + 5*x^2
point = 2  # We want to evaluate this polynomial at x = 2
result = method(coefficients, point)

print(f"The result of the polynomial evaluation at x = {point} is: {result}")
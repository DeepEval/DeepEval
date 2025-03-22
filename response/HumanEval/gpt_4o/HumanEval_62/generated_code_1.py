def method(xs):
    # Create an empty list to store the coefficients of the derivative
    derivative = []
    
    # Iterate over the coefficients and their corresponding powers
    for power, coef in enumerate(xs):
        if power > 0:  # Skip the constant term
            # Calculate the derivative coefficient for this term
            derivative.append(coef * power)
    
    return derivative

# Test case
xs = [3, 4, 5]  # Represents the polynomial 3 + 4x + 5x^2
output = method(xs)  # Should return [4, 10] representing the polynomial 4 + 10x
print(output)
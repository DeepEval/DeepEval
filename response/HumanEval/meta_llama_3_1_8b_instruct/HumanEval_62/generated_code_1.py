def method():
    def polynomial_derivative(xs):
        # Use list comprehension to apply the power rule of differentiation
        return [i * x for i, x in enumerate(xs[1:], start=1)]
    
    xs = [3, 2, 1]  # coefficients of the polynomial 3 + 2x + x^2
    derivative = polynomial_derivative(xs)
    
    # Convert the derivative coefficients to a string for display
    output = " + ".join(f"{x}x^{i}" for i, x in enumerate(derivative, start=1))
    if derivative:
        output = output.replace("+ -", " - ")
    else:
        output = "0"
    
    return output

print(method())  # Output: 2x + x^2
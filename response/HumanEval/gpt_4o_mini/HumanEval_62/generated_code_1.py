def method(xs):
    # Calculate the derivative of the polynomial represented by xs
    derivative = []
    for i in range(1, len(xs)):
        # The derivative coefficient for x^(i-1) is i * xs[i]
        derivative_coefficient = i * xs[i]
        derivative.append(derivative_coefficient)
    
    return derivative

# Test case
if __name__ == "__main__":
    # Example polynomial: 2 + 3x + 4x^2 (represented as [2, 3, 4])
    xs = [2, 3, 4]
    output = method(xs)
    print("The derivative of the polynomial is:", output)
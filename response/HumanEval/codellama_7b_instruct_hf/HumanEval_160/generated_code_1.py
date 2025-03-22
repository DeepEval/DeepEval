from sympy import symbols, sympify, evaluate

def method():
    # Initialize variables
    operator = ["+", "-", "*", "//", "**"]
    operand = [4, 6, 7, 9, 10, 3]

    # Create symbolic variables
    x, y, z, t, u, v = symbols("x y z t u v")

    # Build the algebraic expression
    expression = operand[0]
    for i in range(1, len(operator)):
        expression = operator[i - 1] + "(" + str(expression) + "," + str(operand[i]) + ")"

    # Simplify and evaluate the expression
    simplified_expression = sympify(expression)
    output = evaluate(simplified_expression)

    return output

# Test case
test_case = method()
print(test_case)
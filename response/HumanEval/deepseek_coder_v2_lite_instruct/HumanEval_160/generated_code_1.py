def method():
    import operator

    # Define the basic algebra operations
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '//': operator.floordiv,
        '**': operator.pow
    }

    # Example lists
    operators = ['+', '-', '*', '//', '**']
    operands = [1, 2, 3, 4, 5]

    # Initialize the expression with the first operand
    expression = str(operands[0])

    # Build the expression
    for op, operand in zip(operators, operands[1:]):
        expression += f" {op} {operand}"

    # Evaluate the expression
    result = eval(expression)

    return result

# Test case
output = method()
print(f"The result of the expression is: {output}")
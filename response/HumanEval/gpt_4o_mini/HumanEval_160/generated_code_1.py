def method():
    # Example input
    operators = ['+', '-', '*', '//', '**']
    operands = [2, 3, 4, 2, 1]
    
    # Check if the length of operators and operands are correct
    if len(operators) + 1 != len(operands):
        raise ValueError("Invalid input: The number of operands must be one more than the number of operators.")

    # Build the expression
    expression = str(operands[0])  # Start with the first operand
    for i in range(len(operators)):
        expression += f" {operators[i]} {operands[i + 1]}"
    
    # Evaluate the expression
    output = eval(expression)
    return output

# Test case for validation
result = method()
print("The result of the expression is:", result)
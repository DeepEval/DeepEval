def method(operators, operands):
    # Start with the first operand
    result = operands[0]
    
    # Iterate through the operators and operands
    for i in range(len(operators)):
        operator = operators[i]
        operand = operands[i + 1]
        
        # Perform the operation based on the current operator
        if operator == '+':
            result += operand
        elif operator == '-':
            result -= operand
        elif operator == '*':
            result *= operand
        elif operator == '//':
            result //= operand
        elif operator == '**':
            result **= operand
        else:
            raise ValueError("Unsupported operator: " + operator)
    
    return result

# Test case
operators = ['+', '*', '-', '//', '**']
operands = [5, 3, 2, 7, 2, 3]  # This would correspond to the expression: (((((5 + 3) * 2) - 7) // 2) ** 3)
output = method(operators, operands)
print(output)  # Expected output: 64
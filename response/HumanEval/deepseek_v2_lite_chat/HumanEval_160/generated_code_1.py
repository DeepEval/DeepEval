import operator

def method():
    # Define the basic algebra operations
    operations = [
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.pow
    ]
    
    # Given lists
    operator_list = ['+', '-', '*', '//', '**']
    operand_list = [1, 2, 3, 4, 5]
    
    # Build the algebric expression
    expression = ' '.join([str(x) for x in operand_list])
    for i in range(len(operator_list)):
        expression = expression.replace(str(operator_list[i]), str(operations[i]))
    
    # Evaluate the expression
    output = eval(expression)
    
    return output
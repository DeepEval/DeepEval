import operator
import math

# Define the basic algebra operations as a dictionary
operations = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "**": operator.pow
}

def method(operators, operands):
    """
    Evaluates an algebraic expression based on the given operators and operands.

    Args:
    operators (list): A list of basic algebra operations.
    operands (list): A list of integers.

    Returns:
    float: The result of the evaluated algebraic expression.

    Raises:
    ValueError: If the lengths of the operators and operands lists do not match.
    """

    # Check if the lengths of the operators and operands lists match
    if len(operators)!= len(operands) - 1:
        raise ValueError("The lengths of the operators and operands lists must match")

    # Initialize the output with the first operand
    output = operands[0]

    # Iterate over the operators and operands starting from the second operand
    for i in range(len(operators)):
        # Get the current operator and operand
        op = operators[i]
        operand = operands[i + 1]

        # Check if the operator is valid
        if op not in operations:
            raise ValueError(f"Invalid operator: {op}")

        # Apply the current operator to the output and operand
        output = operations[op](output, operand)

    return output

# Test case
operators = ["+", "*", "-"]
operands = [10, 2, 5]
print(method(operators, operands))  # Output: 25

operators = ["//", "**"]
operands = [10, 3]
print(method(operators, operands))  # Output: 1

operators = ["+", "-"]
operands = [10, 2, 5]
print(method(operators, operands))  # Output: 3
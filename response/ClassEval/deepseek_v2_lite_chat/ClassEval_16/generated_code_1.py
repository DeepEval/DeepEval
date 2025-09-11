class Calculator:
    def __init__(self):
        self.operators = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y
        }

    def calculate(self, expression):
        try:
            # Split the expression by spaces
            tokens = expression.split()
            # Initialize operands and operators stacks
            operand_stack = []
            operator_stack = []
            # Parse the expression
            for token in tokens:
                if token in self.operators:
                    # If the token is an operator, pop from operand stack and operator stack
                    operand1 = operand_stack.pop()
                    operand2 = operand_stack.pop()
                    # Perform the operation and push the result back
                    result = self.operators[token](operand1, operand2)
                    operand_stack.append(result)
                else:
                    operand_stack.append(float(token))
            # Return the result after parsing
            return operand_stack[-1]
        except Exception as e:
            print(f"Error: {e}")
            return None

    def precedence(self, operator):
        priorities = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3
        }
        return priorities.get(operator, 0)

    def apply_operator(self, operand_stack, operator_stack):
        while operator_stack and self.precedence(operator_stack[-1]) >= self.precedence(operator_stack[-1]):
            # Pop operators from operator stack with higher or equal precedence
            operator = operator_stack.pop()
            operand2 = operand_stack.pop()
            operand1 = operand_stack.pop()
            result = self.operators[operator](operand1, operand2)
            operand_stack.append(result)
        operator_stack.append(operator)
        return operand_stack, operator_stack

# Test cases
if __name__ == "__main__":
    calculator = Calculator()

    # Test calculate
    print(calculator.calculate('1+2-3'))  # Expected output: 0.0
    print(calculator.calculate('2*3/4^5'))  # Expected output: 1.0

    # Test precedence
    print(calculator.precedence('+'))  # Expected output: 1
    print(calculator.precedence('^'))  # Expected output: 3

    # Test apply_operator
    operand_stack = [1, 2, 3]
    operator_stack = ['+']
    updated_operand_stack, updated_operator_stack = calculator.apply_operator(operand_stack, operator_stack)
    print(updated_operand_stack)  # Expected output: [1, -1]
    print(updated_operator_stack)  # Expected output: ['+']
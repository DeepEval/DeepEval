class Calculator:
    """
    This is a class for a calculator, capable of performing basic arithmetic calculations on numerical expressions using the operators +, -, *, /, and ^ (exponentiation).
    """

    def __init__(self):
        """
        Initialize the operations performed by the five operators '+', '-', '*', '/', '^'
        """
        self.operators = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y
        }

    def calculate(self, expression):
        """
        Calculate the value of a given expression
        :param expression: string, given expression
        :return: If successful, returns the value of the expression; otherwise, returns None
        """
        def precedence(op):
            if op in ('+', '-'):
                return 1
            if op in ('*', '/'):
                return 2
            if op == '^':
                return 3
            return 0

        def apply_operator(operands, operators):
            right = operands.pop()
            left = operands.pop()
            operator = operators.pop()
            result = self.operators[operator](left, right)
            operands.append(result)

        def is_number(c):
            return c.isdigit() or c == '.'

        operand_stack = []
        operator_stack = []
        i = 0
        while i < len(expression):
            if is_number(expression[i]):
                num = []
                while i < len(expression) and is_number(expression[i]):
                    num.append(expression[i])
                    i += 1
                operand_stack.append(float(''.join(num)))
                continue
            elif expression[i] in self.operators:
                while (operator_stack and precedence(operator_stack[-1]) >= precedence(expression[i])):
                    apply_operator(operand_stack, operator_stack)
                operator_stack.append(expression[i])
            i += 1

        while operator_stack:
            apply_operator(operand_stack, operator_stack)

        return operand_stack[0] if operand_stack else None

    def precedence(self, operator):
        """
        Returns the priority of the specified operator, where the higher the priority, the greater the assignment. The priority of '^' is greater than '/' and '*', and the priority of '/' and '*' is greater than '+' and '-'
        :param operator: string, given operator
        :return: int, the priority of the given operator, otherwise return 0
        """
        if operator in ('+', '-'):
            return 1
        if operator in ('*', '/'):
            return 2
        if operator == '^':
            return 3
        return 0

    def apply_operator(self, operand_stack, operator_stack):
        """
        Use the operator at the top of the operator stack to perform the operation on the two numbers at the top of the operand stack, and store the result back on the operand stack.
        :param operand_stack: list
        :param operator_stack: list
        :return: the updated operand_stack and operator_stack
        """
        right = operand_stack.pop()
        left = operand_stack.pop()
        operator = operator_stack.pop()
        result = self.operators[operator](left, right)
        operand_stack.append(result)
        return operand_stack, operator_stack

if __name__ == "__main__":
    # Test cases
    calculator = Calculator()
    
    # Test calculate method
    output = calculator.calculate('1+2-3')
    print(output)  # Expected output: 0.0
    
    # Test precedence method
    output = calculator.precedence('+')
    print(output)  # Expected output: 1
    
    output = calculator.precedence('^')
    print(output)  # Expected output: 3
    
    # Test apply_operator method
    output = calculator.apply_operator([1, 2, 3], ['+', '-'])
    print(output)  # Expected output: ([1, -1], ['-'])
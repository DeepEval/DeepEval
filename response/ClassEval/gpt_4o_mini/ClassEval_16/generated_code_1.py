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
        >>> calculator = Calculator()
        >>> calculator.calculate('1+2-3')
        0.0
        """
        operand_stack = []
        operator_stack = []
        i = 0
        expression = expression.replace(' ', '')  # Remove spaces for easier processing
        
        while i < len(expression):
            char = expression[i]
            if char.isdigit():  # If the character is a digit
                num = 0
                while i < len(expression) and expression[i].isdigit():
                    num = num * 10 + int(expression[i])
                    i += 1
                operand_stack.append(float(num))
                continue  # Skip the increment of i since it's already handled
            elif char in self.operators:  # If the character is an operator
                while (operator_stack and 
                       self.precedence(operator_stack[-1]) >= self.precedence(char)):
                    self.apply_operator(operand_stack, operator_stack)
                operator_stack.append(char)
            i += 1
        
        while operator_stack:
            self.apply_operator(operand_stack, operator_stack)

        return operand_stack[0] if operand_stack else None

    def precedence(self, operator):
        """
        Returns the priority of the specified operator, where the higher the priority, the greater the assignment. 
        The priority of '^' is greater than '/' and '*', and the priority of '/' and '*' is greater than '+' and '-'
        :param operator: string, given operator
        :return: int, the priority of the given operator, otherwise return 0
        >>> calculator = Calculator()
        >>> calculator.precedence('+')
        1
        >>> calculator.precedence('^')
        3
        """
        if operator == '+':
            return 1
        elif operator == '-':
            return 1
        elif operator == '*':
            return 2
        elif operator == '/':
            return 2
        elif operator == '^':
            return 3
        return 0

    def apply_operator(self, operand_stack, operator_stack):
        """
        Use the operator at the top of the operator stack to perform the operation on the two numbers at the top of the operator stack, 
        and store the results at the top of the operator stack
        :param operand_stack: list
        :param operator_stack: list
        :return: the updated operand_stack and operator_stack
        >>> calculator = Calculator()
        >>> calculator.apply_operator([1, 2, 3], ['+', '-'])
        ([1, -1], ['-'])
        """
        operator = operator_stack.pop()
        right_operand = operand_stack.pop()
        left_operand = operand_stack.pop()
        result = self.operators[operator](left_operand, right_operand)
        operand_stack.append(result)
        return operand_stack, operator_stack


if __name__ == "__main__":
    # Test cases for Calculator class methods
    calculator = Calculator()
    
    # Test calculate method
    output = calculator.calculate('1 + 2 - 3 * 4 / 2 ^ 2')
    print(output)  # Expected output: 0.0
    
    # Test precedence method
    output = calculator.precedence('+')
    print(output)  # Expected output: 1
    output = calculator.precedence('^')
    print(output)  # Expected output: 3
    
    # Test apply_operator method
    operand_stack, operator_stack = calculator.apply_operator([1, 2], ['+'])
    print(operand_stack)  # Expected output: [3.0]
    print(operator_stack)  # Expected output: []
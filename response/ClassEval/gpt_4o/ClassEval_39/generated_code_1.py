from collections import deque
from decimal import Decimal

class ExpressionCalculator:
    """
    This is a class in Python that can perform calculations with basic arithmetic operations, including addition, subtraction, multiplication, division, and modulo.
    """

    def __init__(self):
        """
        Initialize the expression calculator
        """
        self.postfix_stack = deque()
        # '+', '-', '*', '/', '(', ')', '%'
        self.operator_priority = {'+': 1, '-': 1, '*': 2, '/': 2, '%': 2, '(': 0}

    def calculate(self, expression):
        """
        Calculate the result of the given postfix expression
        :param expression: string, the infix expression to be calculated
        :return: float, the calculated result
        """
        self.prepare(expression)
        operands = deque()
        
        for token in self.postfix_stack:
            if token.isdigit():
                operands.append(Decimal(token))
            else:
                second = operands.pop()
                first = operands.pop()
                result = self._calculate(first, second, token)
                operands.append(result)
                
        return float(operands.pop())

    def prepare(self, expression):
        """
        Prepare the infix expression for conversion to postfix notation
        :param expression: string, the infix expression to be prepared
        """
        transformed = self.transform(expression)
        output = []
        operators = deque()
        
        for char in transformed:
            if char.isdigit():
                output.append(char)
            elif char == '(':
                operators.append(char)
            elif char == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()
            else:  # operator
                while operators and self.compare(char, operators[-1]):
                    output.append(operators.pop())
                operators.append(char)
        
        while operators:
            output.append(operators.pop())
        
        self.postfix_stack = output

    @staticmethod
    def is_operator(c):
        """
        Check if a character is an operator in {'+', '-', '*', '/', '(', ')', '%'}
        :param c: string, the character to be checked
        :return: bool, True if the character is an operator, False otherwise
        """
        return c in {'+', '-', '*', '/', '(', ')', '%'}

    def compare(self, cur, peek):
        """
        Compare the precedence of two operators
        :param cur: string, the current operator
        :param peek: string, the operator at the top of the operator stack
        :return: bool, True if the current operator has less or equal precedence, False otherwise
        """
        return self.operator_priority[cur] <= self.operator_priority[peek]

    @staticmethod
    def _calculate(first_value, second_value, current_op):
        """
        Perform the mathematical calculation based on the given operands and operator
        :param first_value: string, the first operand
        :param second_value: string, the second operand
        :param current_op: string, the operator
        :return: decimal.Decimal, the calculated result
        """
        first_value = Decimal(first_value)
        second_value = Decimal(second_value)
        
        if current_op == '+':
            return first_value + second_value
        elif current_op == '-':
            return first_value - second_value
        elif current_op == '*':
            return first_value * second_value
        elif current_op == '/':
            return first_value / second_value
        elif current_op == '%':
            return first_value % second_value

    @staticmethod
    def transform(expression):
        """
        Transform the infix expression to a format suitable for conversion
        :param expression: string, the infix expression to be transformed
        :return: string, the transformed expression
        """
        return expression.replace(' ', '')


if __name__ == "__main__":
    # Test case for calculate
    expression_calculator = ExpressionCalculator()
    output = expression_calculator.calculate("2 + 3 * 4")
    print(output)  # Expected output: 14.0

    # Test case for prepare
    expression_calculator.prepare("2+3*4")
    print(list(expression_calculator.postfix_stack))  # Expected output: ['2', '3', '4', '*', '+']

    # Test case for is_operator
    print(expression_calculator.is_operator("+"))  # Expected output: True

    # Test case for compare
    print(expression_calculator.compare("+", "-"))  # Expected output: True

    # Test case for _calculate
    print(expression_calculator._calculate("2", "3", "+"))  # Expected output: 5.0

    # Test case for transform
    print(expression_calculator.transform("2 + 3 * 4"))  # Expected output: "2+3*4"
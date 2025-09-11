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
        self.operator_stack = deque()
        self.operator_priority = { '+': 1, '-': 1, '*': 2, '/': 2, '%': 2, '(': 0, ')': 0 }

    def calculate(self, expression):
        """
        Calculate the result of the given postfix expression
        :param expression: string, the postfix expression to be calculated
        :return: float, the calculated result
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.calculate("2 + 3 * 4")
        14.0

        """
        self.postfix_stack = deque(expression.split())
        return self._calculate_postfix()

    def prepare(self, expression):
        """
        Prepare the infix expression for conversion to postfix notation
        :param expression: string, the infix expression to be prepared
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.prepare("2+3*4")

        expression_calculator.postfix_stack = ['2', '3', '4', '*', '+']
        """
        self.postfix_stack = self._infix_to_postfix(expression)
        return self.postfix_stack

    @staticmethod
    def is_operator(c):
        """
        Check if a character is an operator in {'+', '-', '*', '/', '(', ')', '%'}
        :param c: string, the character to be checked
        :return: bool, True if the character is an operator, False otherwise
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.is_operator("+")
        True

        """
        return c in {'+', '-', '*', '/', '(', ')', '%'}

    def compare(self, cur, peek):
        """
        Compare the precedence of two operators
        :param cur: string, the current operator
        :param peek: string, the operator at the top of the operator stack
        :return: bool, True if the current operator has higher or equal precedence, False otherwise
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.compare("+", "-")
        True

        """
        return self.operator_priority[cur] >= self.operator_priority[peek]

    @staticmethod
    def _calculate(first_value, second_value, current_op):
        """
        Perform the mathematical calculation based on the given operands and operator
        :param first_value: string, the first operand
        :param second_value: string, the second operand
        :param current_op: string, the operator
        :return: decimal.Decimal, the calculated result
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator._calculate("2", "3", "+")
        5.0

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
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.transform("2 + 3 * 4")
        "2+3*4"

        """
        return expression

    def _infix_to_postfix(self, expression):
        """
        Convert infix expression to postfix notation
        :param expression: string, the infix expression to be converted
        :return: deque, the postfix expression
        """
        tokens = []
        output = deque()
        operator_stack = deque()

        for char in expression:
            if not self.is_operator(char):
                tokens.append(char)
            elif char == '(':
                operator_stack.append(char)
            elif char == ')':
                while operator_stack and operator_stack[-1]!= '(':
                    output.append(operator_stack.pop())
                operator_stack.pop()
            else:
                while operator_stack and self.compare(char, operator_stack[-1]):
                    output.append(operator_stack.pop())
                operator_stack.append(char)

        while operator_stack:
            output.append(operator_stack.pop())

        return output

    def _calculate_postfix(self):
        """
        Calculate the result of the postfix expression
        :return: float, the calculated result
        """
        output = []
        for token in self.postfix_stack:
            if not self.is_operator(token):
                output.append(Decimal(token))
            else:
                right = output.pop()
                left = output.pop()
                result = self._calculate(str(left), str(right), token)
                output.append(result)
        return output[0]

if __name__ == "__main__":
    expression_calculator = ExpressionCalculator()
    expression = "2 3 4 * +"
    result = expression_calculator.calculate(expression)
    print(f"Test case for calculate method: {expression} -> {result}")

    expression = "2+3*4"
    output = expression_calculator.prepare(expression)
    print(f"Test case for prepare method: {expression} -> {output}")

    print(f"Test case for is_operator method: is_operator '+' -> {expression_calculator.is_operator('+')}")

    print(f"Test case for compare method: compare '+' and '-' -> {expression_calculator.compare('+', '-')}")

    print(f"Test case for _calculate method: _calculate '2', '3', '+' -> {expression_calculator._calculate('2', '3', '+')}")

    expression = "2 + 3 * 4"
    print(f"Test case for transform method: transform '{expression}' -> '{expression_calculator.transform(expression)}'")
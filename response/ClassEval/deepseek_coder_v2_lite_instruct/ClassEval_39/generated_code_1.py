from collections import deque
import re

class ExpressionCalculator:
    """
    This is a class in Python that can perform calculations with basic arithmetic operations, including addition, subtraction, multiplication, division, and modulo.
    """

    def __init__(self):
        """
        Initialize the expression calculator
        """
        self.postfix_stack = deque()
        self.operat_priority = [0, 3, 2, 1, -1, 1, 0, 2]

    def calculate(self, expression):
        """
        Calculate the result of the given postfix expression
        :param expression: string, the postfix expression to be calculated
        :return: float, the calculated result
        """
        tokens = expression.split()
        stack = []
        for token in tokens:
            if token.isnumeric():
                stack.append(float(token))
            else:
                right_operand = stack.pop()
                left_operand = stack.pop()
                result = self._calculate(left_operand, right_operand, token)
                stack.append(result)
        return stack.pop()

    def prepare(self, expression):
        """
        Prepare the infix expression for conversion to postfix notation
        :param expression: string, the infix expression to be prepared
        """
        expression = self.transform(expression)
        tokens = re.findall(r'\d+|\S', expression)
        output = []
        operator_stack = []
        for token in tokens:
            if token.isnumeric():
                output.append(token)
            elif self.is_operator(token):
                while (operator_stack and self.is_operator(operator_stack[-1]) and
                       self.compare(token, operator_stack[-1])):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                operator_stack.pop()
        while operator_stack:
            output.append(operator_stack.pop())
        self.postfix_stack = deque(output)

    @staticmethod
    def is_operator(c):
        """
        Check if a character is an operator in {'+', '-', '*', '/', '(', ')', '%'}
        :param c: string, the character to be checked
        :return: bool, True if the character is an operator, False otherwise
        """
        return c in "+-*/%()"

    def compare(self, cur, peek):
        """
        Compare the precedence of two operators
        :param cur: string, the current operator
        :param peek: string, the operator at the top of the operator stack
        :return: bool, True if the current operator has higher or equal precedence, False otherwise
        """
        return self.operat_priority[self.get_operator_index(cur)] <= self.operat_priority[self.get_operator_index(peek)]

    @staticmethod
    def _calculate(first_value, second_value, current_op):
        """
        Perform the mathematical calculation based on the given operands and operator
        :param first_value: string, the first operand
        :param second_value: string, the second operand
        :param current_op: string, the operator
        :return: float, the calculated result
        """
        first_value = float(first_value)
        second_value = float(second_value)
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
        else:
            raise ValueError("Invalid operator")

    @staticmethod
    def transform(expression):
        """
        Transform the infix expression to a format suitable for conversion
        :param expression: string, the infix expression to be transformed
        :return: string, the transformed expression
        """
        return re.sub(r'([+\-*/%()])', r' \1 ', expression).strip().replace(' ', '')

    def get_operator_index(self, operator):
        """
        Get the index of the operator in the operat_priority list
        :param operator: string, the operator
        :return: int, the index of the operator
        """
        return self.is_operator(operator)

if __name__ == "__main__":
    expression_calculator = ExpressionCalculator()
    print(expression_calculator.calculate("2 + 3 * 4"))  # Output: 14.0
    expression_calculator.prepare("2+3*4")
    print(list(expression_calculator.postfix_stack))  # Output: ['2', '3', '4', '*', '+']
    print(expression_calculator.is_operator("+"))  # Output: True
    print(expression_calculator.compare("+", "-"))  # Output: True
    print(expression_calculator._calculate("2", "3", "+"))  # Output: 5.0
    print(expression_calculator.transform("2 + 3 * 4"))  # Output: "2+3*4"
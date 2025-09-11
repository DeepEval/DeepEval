from collections import deque

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
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.calculate("2 + 3 * 4")
        14.0

        """
        for token in expression.split():
            if token.isdigit():
                self.postfix_stack.append(float(token))
            else:
                operand2 = self.postfix_stack.pop()
                operand1 = self.postfix_stack.pop()
                result = self._calculate(operand1, operand2, token)
                self.postfix_stack.append(result)
        return self.postfix_stack.pop()

    def prepare(self, expression):
        """
        Prepare the infix expression for conversion to postfix notation
        :param expression: string, the infix expression to be prepared
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.prepare("2+3*4")

        expression_calculator.postfix_stack = ['2', '3', '4', '*', '+']
        """
        operator_stack = []
        output = []
        for token in expression:
            if token.isdigit():
                output.append(token)
            elif token in ['+', '-', '*', '/', '%', '(']:
                while operator_stack and self.compare(token, operator_stack[-1]):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                operator_stack.pop()  # Remove the '('
        while operator_stack:
            output.append(operator_stack.pop())
        self.postfix_stack = deque(output)

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
        return self.operat_priority[self.get_priority(cur)] >= self.operat_priority[self.get_priority(peek)]

    def get_priority(self, op):
        """
        Get the precedence of an operator
        :param op: string, the operator
        :return: int, the precedence of the operator
        """
        return {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '%': 2,
            '(': -1
        }.get(op, 0)

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



if __name__ == "__main__":
    instance = ExpressionCalculator()
    output = instance.calculate("2 3 4 * +")
    print(f"Calculate: {output}")

    instance.prepare("2+3*4")
    print(f"Prepare: {list(instance.postfix_stack)}")

    output = instance.is_operator("+")
    print(f"is_operator: {output}")

    output = instance.compare("+", "-")
    print(f"compare: {output}")

    output = instance._calculate(2, 3, "+")
    print(f"_calculate: {output}")

    output = instance.transform("2 + 3 * 4")
    print(f"transform: {output}")
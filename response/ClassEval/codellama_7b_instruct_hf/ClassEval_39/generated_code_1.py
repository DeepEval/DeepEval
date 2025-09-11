import decimal
from collections import deque

class ExpressionCalculator:
    def __init__(self):
        self.postfix_stack = deque()
        self.operat_priority = [0, 3, 2, 1, -1, 1, 0, 2]

    def calculate(self, expression):
        self.prepare(expression)
        return self._calculate_postfix(self.postfix_stack)

    def prepare(self, expression):
        # Transform the infix expression to postfix notation
        infix_to_postfix = {
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
            "%": "%",
            "(": "(",
            ")": ")",
        }
        for token in expression:
            if token in infix_to_postfix:
                self.postfix_stack.append(infix_to_postfix[token])
            else:
                self.postfix_stack.append(token)

    @staticmethod
    def is_operator(c):
        return c in "+-*/%("

    def compare(self, cur, peek):
        if cur in "+-" and peek in "+-":
            return self.operat_priority[cur] >= self.operat_priority[peek]
        elif cur in "*/" and peek in "*/" and self.operat_priority[cur] >= self.operat_priority[peek]:
            return True
        else:
            return False

    @staticmethod
    def _calculate(first_value, second_value, current_op):
        if current_op == "+":
            return decimal.Decimal(first_value) + decimal.Decimal(second_value)
        elif current_op == "-":
            return decimal.Decimal(first_value) - decimal.Decimal(second_value)
        elif current_op == "*":
            return decimal.Decimal(first_value) * decimal.Decimal(second_value)
        elif current_op == "/":
            return decimal.Decimal(first_value) / decimal.Decimal(second_value)
        else:
            return decimal.Decimal(first_value) % decimal.Decimal(second_value)

    @staticmethod
    def _calculate_postfix(postfix_stack):
        # Evaluate the postfix expression using the Shunting-yard algorithm
        operand_stack = deque()
        while postfix_stack:
            token = postfix_stack.popleft()
            if token in "+-*/%(":
                if token == "(":
                    operand_stack.append(token)
                elif token == ")":
                    while operand_stack[-1] != "(":
                        first_value = decimal.Decimal(operand_stack.pop())
                        second_value = decimal.Decimal(operand_stack.pop())
                        current_op = operand_stack.pop()
                        result = self._calculate(first_value, second_value, current_op)
                        operand_stack.append(result)
                    operand_stack.pop()
                else:
                    first_value = decimal.Decimal(operand_stack.pop())
                    second_value = decimal.Decimal(operand_stack.pop())
                    current_op = token
                    result = self._calculate(first_value, second_value, current_op)
                    operand_stack.append(result)
            else:
                operand_stack.append(token)
        return operand_stack.pop()

if __name__ == "__main__":
    expression = "2 + 3 * 4"
    calculator = ExpressionCalculator()
    result = calculator.calculate(expression)
    print(result)
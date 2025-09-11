from collections import deque
from decimal import Decimal

class ExpressionCalculator:
    def __init__(self):
        self.postfix_stack = deque()
        self.operat_priority = [0, 3, 2, 1, -1, 1, 0, 2]

    def calculate(self, postfix_expression):
        tokens = postfix_expression.split()
        stack = []
        for token in tokens:
            if token in self.operat_priority:
                op1 = stack.pop()
                op2 = stack.pop()
                result = self._calculate(float(op1), float(op2), token)
                stack.append(str(result))
            else:
                stack.append(token)
        return float(stack[0])

    def prepare(self, infix_expression):
        self.postfix_stack = deque()
        operator_stack = []
        for token in infix_expression:
            if token.isdigit():
                self.postfix_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    self.postfix_stack.append(operator_stack.pop())
                operator_stack.pop()  # remove '('
            else:
                while (operator_stack and operator_stack[-1] in self.operat_priority and
                       self.operat_priority[self.operat_priority.index(operator_stack[-1])]
                       >= self.operat_priority[self.operat_priority.index(token)]):
                    self.postfix_stack.append(operator_stack.pop())
                operator_stack.append(token)
        while operator_stack:
            self.postfix_stack.append(operator_stack.pop())
        return self.postfix_stack

    @staticmethod
    def is_operator(c):
        return c in {'+', '-', '*', '/', '(', ')'}

    def compare(self, cur, peek):
        return self.operat_priority.index(cur) >= self.operat_priority.index(peek)

    @staticmethod
    def _calculate(first_value, second_value, current_op):
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
        return expression

# Tests
if __name__ == "__main__":
    calculator = ExpressionCalculator()
    
    # Test calculate method
    print(calculator.calculate("2 + 3 * 4"))  # Output: 14.0
    
    # Test prepare method
    prepared_expression = calculator.prepare("2 + 3 * 4")
    
    # Test transform method
    transformed_expression = ExpressionCalculator.transform("2 + 3 * 4")

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
        operand_stack = []
        operator_stack = []
        for token in expression.split():
            if token in self.operators:
                self.apply_operator(operand_stack, operator_stack)
                operator_stack.append(token)
            else:
                operand_stack.append(float(token))
        self.apply_operator(operand_stack, operator_stack)
        return operand_stack.pop()

    def apply_operator(self, operand_stack, operator_stack):
        if not operand_stack or not operator_stack:
            return
        operator = operator_stack.pop()
        operand1 = operand_stack.pop()
        operand2 = operand_stack.pop()
        result = self.operators[operator](operand1, operand2)
        operand_stack.append(result)

    def precedence(self, operator):
        precedence = 0
        if operator in self.operators:
            precedence = self.operators[operator].precedence
        return precedence

if __name__ == "__main__":
    calculator = Calculator()
    expression = "1 + 2 - 3"
    result = calculator.calculate(expression)
    print(result)
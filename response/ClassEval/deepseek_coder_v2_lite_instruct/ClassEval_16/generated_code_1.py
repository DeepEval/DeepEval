class Calculator:
    """
    This is a class for a calculator, capable of performing basic arithmetic calculations on numerical expressions using the operators +, -, *, /, and ^ (exponentiation).
    """

    def __init__(self):
        """
        Initialize the operations performed by the five operators '+','-','*','/','^'
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
        try:
            tokens = self.tokenize(expression)
            postfix = self.infix_to_postfix(tokens)
            result = self.evaluate_postfix(postfix)
            return result
        except Exception as e:
            print(f"Error: {e}")
            return None

    def tokenize(self, expression):
        """
        Tokenize the expression into numbers and operators.
        """
        import re
        tokens = []
        token_pattern = re.compile(r'\d+\.?\d*|[+\-*/^()]')
        for match in token_pattern.finditer(expression):
            tokens.append(match.group())
        return tokens

    def precedence(self, operator):
        """
        Returns the priority of the specified operator, where the higher the priority, the greater the assignment. The priority of '^' is greater than '/' and '*', and the priority of '/' and '*' is greater than '+' and '-'
        :param operator: string, given operator
        :return: int, the priority of the given operator, otherwise return 0
        """
        precedence_map = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        return precedence_map.get(operator, 0)

    def infix_to_postfix(self, tokens):
        """
        Convert infix expression to postfix expression.
        """
        output = []
        operators = []
        for token in tokens:
            if token.isdigit() or '.' in token:
                output.append(token)
            elif token in self.operators:
                while (operators and operators[-1] != '(' and
                       self.precedence(operators[-1]) >= self.precedence(token)):
                    output.append(operators.pop())
                operators.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # Remove '('
        while operators:
            output.append(operators.pop())
        return output

    def evaluate_postfix(self, tokens):
        """
        Evaluate a postfix expression.
        """
        stack = []
        for token in tokens:
            if token.isdigit() or '.' in token:
                stack.append(float(token))
            elif token in self.operators:
                y = stack.pop()
                x = stack.pop()
                result = self.operators[token](x, y)
                stack.append(result)
        return stack[0]

    def apply_operator(self, operand_stack, operator_stack):
        """
        Use the operator at the top of the operator stack to perform the operation on the two numbers at the top of the operator stack, and store the results at the top of the operator stack
        :param operand_stack: list
        :param operator_stack: list
        :return: the updated operand_stack and operator_stack
        """
        y = operand_stack.pop()
        x = operand_stack.pop()
        operator = operator_stack.pop()
        result = self.operators[operator](x, y)
        operand_stack.append(result)
        return operand_stack, operator_stack


if __name__ == "__main__":
    calculator = Calculator()
    
    # Test cases
    print(calculator.calculate('1+2-3'))  # Expected output: 0.0
    print(calculator.calculate('3+4*2/(1-5)^2^3'))  # Expected output: 3.0
    print(calculator.calculate('10+2*3'))  # Expected output: 16.0
    print(calculator.calculate('(1+2)*3'))  # Expected output: 9.0
    print(calculator.calculate('10+2*3+4'))  # Expected output: 16.0
    print(calculator.calculate('10+2*3+4^2'))  # Expected output: 26.0

    # Test precedence method
    print(calculator.precedence('+'))  # Expected output: 1
    print(calculator.precedence('-'))  # Expected output: 1
    print(calculator.precedence('*'))  # Expected output: 2
    print(calculator.precedence('/'))  # Expected output: 2
    print(calculator.precedence('^'))  # Expected output: 3

    # Test apply_operator method
    print(calculator.apply_operator([1, 2, 3], ['+', '-']))  # Expected output: ([-1], ['-'])
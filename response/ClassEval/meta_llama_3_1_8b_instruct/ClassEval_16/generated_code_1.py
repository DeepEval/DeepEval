import operator as op

class Calculator:
    """
    This is a class for a calculator, capable of performing basic arithmetic calculations on numerical expressions using the operators +, -, *, /, and ^ (exponentiation).
    """

    def __init__(self):
        """
        Initialize the operations performed by the five operators'+','-','*','/','^'
        """
        self.operators = {
            '+': op.add,
            '-': op.sub,
            '*': op.mul,
            '/': op.truediv,
            '^': op.pow
        }

    def calculate(self, expression):
        """
        Calculate the value of a given expression
        :param expression: string, given expression
        :return:If successful, returns the value of the expression; otherwise, returns None
        >>> calculator = Calculator()
        >>> calculator.calculate('1+2-3')
        0.0
        """
        try:
            return eval(expression)
        except Exception as e:
            print(f"Failed to calculate the expression: {str(e)}")
            return None

    def precedence(self, operator):
        """
        Returns the priority of the specified operator, where the higher the priority, the greater the assignment. The priority of '^' is greater than '/' and '*', and the priority of '/' and '*' is greater than '+' and '-'
        :param operator: string, given operator
        :return: int, the priority of the given operator, otherwise return 0
        >>> calculator = Calculator()
        >>> calculator.precedence('+')
        1
        >>> calculator.precedence('^')
        3
        """
        if operator in ['+', '-']:
            return 1
        elif operator in ['*', '/']:
            return 2
        elif operator == '^':
            return 3
        else:
            return 0

    def apply_operator(self, operand_stack, operator_stack):
        """
        Use the operator at the top of the operator stack to perform the operation on the two numbers at the top of the operator stack, and store the results at the top of the operator stack
        :param operand_stack:list
        :param operator_stack:list
        :return: the updated operand_stack and operator_stack
        >>> calculator = Calculator()
        >>> calculator.apply_operator([1, 2, 3], ['+', '-'])
        ([1, -1], ['-'])
        """
        op = operator_stack.pop()
        operand2 = operand_stack.pop()
        operand1 = operand_stack.pop()
        if op == '^':
            operand_stack.append(operand1 ** operand2)
        else:
            operand_stack.append(self.operators[op](operand1, operand2))
        return operand_stack, operator_stack


if __name__ == "__main__":
    calculator = Calculator()
    # Test case for calculate method
    output = calculator.calculate('1+2-3')
    print(f"Output for calculate method: {output}")

    # Test case for precedence method
    output = calculator.precedence('+')
    print(f"Output for precedence method: {output}")

    # Test case for apply_operator method
    operand_stack = [1, 2, 3]
    operator_stack = ['+', '-']
    output = (calculator.apply_operator(operand_stack, operator_stack))
    print(f"Output for apply_operator method: {output}")
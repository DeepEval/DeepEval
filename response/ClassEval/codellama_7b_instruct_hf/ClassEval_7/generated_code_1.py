import re

class BalancedBrackets:
    """
    This is a class that checks for bracket matching
    """

    def __init__(self, expr):
        """
        Initializes the class with an expression.
        :param expr: The expression to check for balanced brackets,str.
        """
        self.stack = []
        self.left_brackets = ["(", "{", "["]
        self.right_brackets = [")", "}", "]"]
        self.expr = expr

    def clear_expr(self):
        """
        Clears the expression of all characters that are not brackets.
        >>> b = BalancedBrackets("a(b)c")
        >>> b.clear_expr()
        >>> b.expr
        '()'

        """
        self.expr = re.sub(r'[^()\[\]{}]', '', self.expr)

    def check_balanced_brackets(self):
        """
        Checks if the expression has balanced brackets.
        :return: True if the expression has balanced brackets, False otherwise.
        >>> b = BalancedBrackets("a(b)c")
        >>> b.check_balanced_brackets()
        True

        """
        if len(self.expr) % 2 == 1:
            return False
        for i in range(len(self.expr) // 2):
            if self.expr[i] in self.left_brackets:
                self.stack.append(self.expr[i])
            elif self.expr[i] in self.right_brackets:
                if self.stack and self.stack[-1] == self.left_brackets[self.right_brackets.index(self.expr[i])]:
                    self.stack.pop()
                else:
                    return False
        return not self.stack

if __name__ == "__main__":
    b = BalancedBrackets("a(b)c")
    print(b.check_balanced_brackets())
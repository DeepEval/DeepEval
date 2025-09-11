class BalancedBrackets:
    """
    This is a class that checks for bracket matching
    """

    def __init__(self, expr):
        """
        Initializes the class with an expression.
        :param expr: The expression to check for balanced brackets, str.
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
        self.expr = ''.join([bracket for bracket in self.expr if bracket in self.left_brackets])

    def check_balanced_brackets(self):
        """
        Checks if the expression has balanced brackets.
        :return: True if the expression has balanced brackets, False otherwise.
        """
        for bracket in self.expr:
            if bracket in self.left_brackets:
                self.stack.append(bracket)
            elif bracket in self.right_brackets:
                if not self.stack or self.left_brackets.index(self.stack[-1]) != self.right_brackets.index(bracket):
                    return False
                self.stack.pop()
        return not self.stack

# Test cases
if __name__ == "__main__":
    # Test clear_expr
    b = BalancedBrackets("a(b)c(d)e)f{g)h")
    b.clear_expr()
    print(b.expr == "()()()()")

    # Test check_balanced_brackets
    b = BalancedBrackets("([]{})")
    print(b.check_balanced_brackets())  # Should print True

    b = BalancedBrackets("([)]{}")
    print(b.check_balanced_brackets())  # Should print False
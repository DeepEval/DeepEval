class BalancedBrackets:
    """
    This is a class that checks for bracket matching.
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
        self.expr = ''.join(c for c in self.expr if c in self.left_brackets + self.right_brackets)

    def check_balanced_brackets(self):
        """
        Checks if the expression has balanced brackets.
        :return: True if the expression has balanced brackets, False otherwise.
        >>> b = BalancedBrackets("a(b)c")
        >>> b.check_balanced_brackets()
        True
        """
        self.clear_expr()
        for char in self.expr:
            if char in self.left_brackets:
                self.stack.append(char)
            elif char in self.right_brackets:
                if not self.stack:
                    return False
                top = self.stack.pop()
                if self.left_brackets.index(top) != self.right_brackets.index(char):
                    return False
        return not self.stack

if __name__ == "__main__":
    # Test case for clear_expr method
    instance = BalancedBrackets("a(b)c")
    instance.clear_expr()
    output = instance.expr
    print(f"Cleared expression: {output}")  # Expected output: '()'

    # Test case for check_balanced_brackets method
    instance = BalancedBrackets("a(b)c")
    output = instance.check_balanced_brackets()
    print(f"Balanced brackets: {output}")  # Expected output: True

    # Additional test cases
    instance = BalancedBrackets("([{}])")
    output = instance.check_balanced_brackets()
    print(f"Balanced brackets: {output}")  # Expected output: True

    instance = BalancedBrackets("([)]")
    output = instance.check_balanced_brackets()
    print(f"Balanced brackets: {output}")  # Expected output: False

    instance = BalancedBrackets("((()))")
    output = instance.check_balanced_brackets()
    print(f"Balanced brackets: {output}")  # Expected output: True

    instance = BalancedBrackets("((())")
    output = instance.check_balanced_brackets()
    print(f"Balanced brackets: {output}")  # Expected output: False
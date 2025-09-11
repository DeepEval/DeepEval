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
        self.expr = ''.join([char for char in self.expr if char in self.left_brackets or char in self.right_brackets])

    def check_balanced_brackets(self):
        """
        Checks if the expression has balanced brackets.
        :return: True if the expression has balanced brackets, False otherwise.
        >>> b = BalancedBrackets("a(b)c")
        >>> b.check_balanced_brackets()
        True

        """
        for char in self.expr:
            if char in self.left_brackets:
                self.stack.append(char)
            elif char in self.right_brackets:
                if not self.stack or self.left_brackets.index(self.stack.pop()) != self.right_brackets.index(char):
                    return False
        return not self.stack

if __name__ == "__main__":
    instance = BalancedBrackets("a(b)c")
    instance.clear_expr()
    print(instance.expr)  # Output: ()
    print(instance.check_balanced_brackets())  # Output: True
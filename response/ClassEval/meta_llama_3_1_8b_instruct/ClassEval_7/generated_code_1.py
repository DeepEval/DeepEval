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
        self.expr = ''.join([char for char in self.expr if char in self.left_brackets + self.right_brackets])

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
                if len(self.stack) == 0:
                    return False
                bracket_index = self.right_brackets.index(char)
                if self.left_brackets[bracket_index]!= self.stack.pop():
                    return False
        return len(self.stack) == 0


if __name__ == "__main__":
    # Test cases
    instance = BalancedBrackets("a(b)c")
    instance.clear_expr()
    print(instance.expr)  # Output: '()'

    instance2 = BalancedBrackets("a(b)c")
    print(instance2.check_balanced_brackets())  # Output: True

    instance3 = BalancedBrackets("(b)c")
    print(instance3.check_balanced_brackets())  # Output: False

    instance4 = BalancedBrackets("{[]}")
    print(instance4.check_balanced_brackets())  # Output: True

    instance5 = BalancedBrackets("{[}]")
    print(instance5.check_balanced_brackets())  # Output: False
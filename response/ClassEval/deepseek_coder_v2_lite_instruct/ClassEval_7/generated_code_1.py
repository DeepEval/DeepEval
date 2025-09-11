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
        temp_expr = ""
        for char in self.expr:
            if char in self.left_brackets + self.right_brackets:
                temp_expr += char
        self.expr = temp_expr

    def check_balanced_brackets(self):
        """
        Checks if the expression has balanced brackets.
        :return: True if the expression has balanced brackets, False otherwise.
        >>> b = BalancedBrackets("a(b)c")
        >>> b.check_balanced_brackets()
        True
        """
        bracket_pairs = {")": "(", "}": "{", "]": "["}
        for char in self.expr:
            if char in self.left_brackets:
                self.stack.append(char)
            elif char in self.right_brackets:
                if not self.stack or self.stack[-1] != bracket_pairs[char]:
                    return False
                self.stack.pop()
        return not self.stack


if __name__ == "__main__":
    # Test cases
    b = BalancedBrackets("a(b)c")
    b.clear_expr()
    print(b.expr)  # Expected output: '()'

    b = BalancedBrackets("a(b)c")
    print(b.check_balanced_brackets())  # Expected output: True

    b = BalancedBrackets("a(b{c}d)e")
    b.clear_expr()
    print(b.expr)  # Expected output: '(b{c}d)'

    b = BalancedBrackets("a(b{c}d)e")
    print(b.check_balanced_brackets())  # Expected output: True

    b = BalancedBrackets("a(b{c}d]e")
    b.clear_expr()
    print(b.expr)  # Expected output: '(b{c}d]'

    b = BalancedBrackets("a(b{c}d]e")
    print(b.check_balanced_brackets())  # Expected output: False
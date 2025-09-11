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
        self.expr = ''.join([char for char in self.expr if char in self.left_brackets + self.right_brackets])

    def check_balanced_brackets(self):
        """
        Checks if the expression has balanced brackets.
        :return: True if the expression has balanced brackets, False otherwise.
        >>> b = BalancedBrackets("a(b)c")
        >>> b.check_balanced_brackets()
        True
        >>> b = BalancedBrackets("a(b{c})")
        >>> b.check_balanced_brackets()
        True
        >>> b = BalancedBrackets("a(b[c})")
        >>> b.check_balanced_brackets()
        False
        >>> b = BalancedBrackets("a(b{c)")
        >>> b.check_balanced_brackets()
        False
        """
        self.clear_expr()  # Clear the expression first
        
        for char in self.expr:
            if char in self.left_brackets:
                self.stack.append(char)
            elif char in self.right_brackets:
                if not self.stack:
                    return False
                top = self.stack.pop()
                if self.left_brackets.index(top) != self.right_brackets.index(char):
                    return False
        
        return len(self.stack) == 0


# Test cases
if __name__ == "__main__":
    # Test case for clear_expr method
    b1 = BalancedBrackets("a(b)c")
    b1.clear_expr()
    print(b1.expr)  # Expected output: '()'

    # Test case for check_balanced_brackets method
    b2 = BalancedBrackets("a(b)c")
    print(b2.check_balanced_brackets())  # Expected output: True
    
    b3 = BalancedBrackets("a(b{c})")
    print(b3.check_balanced_brackets())  # Expected output: True
    
    b4 = BalancedBrackets("a(b[c})")
    print(b4.check_balanced_brackets())  # Expected output: False
    
    b5 = BalancedBrackets("a(b{c)")
    print(b5.check_balanced_brackets())  # Expected output: False
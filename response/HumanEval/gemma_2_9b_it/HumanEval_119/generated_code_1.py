def method(str1, str2):
    """
    Checks if it is possible to concatenate two strings of parentheses
    in some order to form a balanced string.

    Args:
        str1: The first string of parentheses.
        str2: The second string of parentheses.

    Returns:
        True if it's possible to concatenate the strings to form a balanced string,
        False otherwise.
    """

    stack = []

    def is_balanced(s):
      for char in s:
        if char == '(':
          stack.append(char)
        elif char == ')':
          if not stack or stack.pop() != '(':
            return False
      return len(stack) == 0

    return is_balanced(str1 + str2) or is_balanced(str2 + str1)

# Test Case
str1 = "()()"
str2 = "(()"
output = method(str1, str2)
print(f"Can concatenate '{str1}' and '{str2}' to form a balanced string? {output}")
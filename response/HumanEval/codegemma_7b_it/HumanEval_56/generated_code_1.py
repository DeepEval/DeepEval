import unittest

def brackets_match(brackets):
    stack = []

    for bracket in brackets:
        if bracket == "<":
            stack.append(bracket)
        elif bracket == ">":
            if not stack or stack.pop() != "<":
                return False

    return not stack

# class TestBrackets(unittest.TestCase):
#     def test_matching_brackets(self):
#         self.assertEqual(brackets_match("<hello>"), True)

#     def test_mismatched_brackets(self):
#         self.assertEqual(brackets_match("<hello>"), True)

#     def test_unmatched_opening_bracket(self):
#         self.assertEqual(brackets_match("<hello>"), False)

#     def test_unmatched_closing_bracket(self):
#         self.assertEqual(brackets_match("<hello>"), False)

# if __name__ == "__main__":
#     unittest.main()

print(brackets_match("<hello>"))

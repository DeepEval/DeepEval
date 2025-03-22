def method():
    def min_additional_parentheses(s):
        # Recursive function to find the minimum number of additional parentheses
        # needed to make the string balanced.
        if not s:
            return 0
        if s[0] == ')' or s[-1] == '(':
            # If the first or last character is not balanced, add 1 to find the minimum.
            return 1 + min_additional_parentheses(s[1:-1])
        else:
            # If the string is balanced, find the minimum additional parentheses needed
            # by looking at the rest of the string.
            return min(min_additional_parentheses(s[:i] + s[i+1:]),
                       min_additional_parentheses(s[:i] + '()' + s[i+1:]))

    # Get the two input strings
    # string1 = input().strip()
    # string2 = input().strip()
    string1 = "(()))("
    string2 = "())(()"

    # Calculate the minimum additional parentheses needed for each string
    min1 = min_additional_parentheses(string1)
    min2 = min_additional_parentheses(string2)

    # Check if it is possible to concatenate the strings
    if min1 + min2 > len(string1) or min1 + min2 > len(string2):
        return "It is not possible to concatenate the strings to form a balanced string."
    else:
        return "It is possible to concatenate the strings to form a balanced string."

# Example test case
print(method())
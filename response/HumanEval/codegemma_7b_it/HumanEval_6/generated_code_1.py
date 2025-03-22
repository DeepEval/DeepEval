from collections import defaultdict

def method():
    # input_str = input("Enter the string: ")
    input_str = "(()())"
    parentheses_dict = defaultdict(int)
    max_depth = 0

    for char in input_str:
        if char == "(":
            parentheses_dict["depth"] += 1
        elif char == ")":
            parentheses_dict["depth"] -= 1
        max_depth = max(max_depth, parentheses_dict["depth"])

    return max_depth

# test_case = "(()())"
output = method()
print(output)
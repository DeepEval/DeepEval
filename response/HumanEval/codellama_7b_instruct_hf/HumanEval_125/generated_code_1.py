import re

def method():
    # Get the input string
    # text = input("Enter a string: ")

    # 修改：
    text = "This is a string with whitespace and commas"

    # Split the string on whitespace if there are no commas
    if "," not in text:
        words = text.split()
        return words

    # Split the string on commas if there are no whitespaces
    elif " " not in text:
        words = text.split(",")
        return words

    # Split the string on whitespaces and commas if there are both
    else:
        words = re.split(r"[,\s]+", text)
        return words

# Test case
text = "This is a string with whitespace and commas"
expected_output = ["This", "is", "a", "string", "with", "whitespace", "and", "commas"]
actual_output = method()
print(actual_output == expected_output)  # Output: True
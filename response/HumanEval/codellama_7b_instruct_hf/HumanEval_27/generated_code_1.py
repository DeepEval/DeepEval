import re

def method():
    input_str = "aBcDeF"
    output = ""

    for char in input_str:
        if re.match(r"[a-z]", char):
            output += char.upper()
        elif re.match(r"[A-Z]", char):
            output += char.lower()
        else:
            output += char

    return output

# Test Case
test_case = "aBcDeF"
actual_output = method()

if actual_output == "AbCdEf":
    print("Test Case Passed")
else:
    print("Test Case Failed")
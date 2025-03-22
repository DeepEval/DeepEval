def method(input_string):
    output = ""
    for char in input_string:
        if char.islower():
            output += char.upper()
        elif char.isupper():
            output += char.lower()
        else:
            output += char
    return output

# Test case
input_string = "Hello World!"
expected_output = "hELLO wORLD!"
# assert method(input_string) == expected_output, f"Test failed! Expected {expected_output}, but got {method(input_string)}"

# print("Test passed!")

method(input_string)
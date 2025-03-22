import sys

def method():
    # Get input from the user
    input_string = sys.argv[1]

    # Return the length of the input string
    return len(input_string)

# Test case
input_string = "Hello, world!"
expected_output = 13

output = method()

if output == expected_output:
    print("Test case passed")
else:
    print("Test case failed")
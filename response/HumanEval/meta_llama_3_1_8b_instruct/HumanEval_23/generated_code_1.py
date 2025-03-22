# Import necessary packages
import builtins

def method(input_string):
    """
    Returns the length of the given string.

    Args:
        input_string (str): The input string.

    Returns:
        int: The length of the input string.
    """
    # Use the built-in len() function to get the length of the string
    output = len(input_string)
    return output

# Test case for validation
input_string = "Hello, World!"
output = method(input_string)
print("Input String:", input_string)
print("Output (Length):", output)

# Expected output:
# Input String: Hello, World!
# Output (Length): 13
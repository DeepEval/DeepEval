def method(input_string):
    """
    This function calculates the sum of ASCII codes of upper case characters in a given string.

    Args:
        input_string (str): The input string.

    Returns:
        int: The sum of ASCII codes of upper case characters.
    """
    # Initialize the sum to 0
    ascii_sum = 0
    
    # Iterate over each character in the string
    for char in input_string:
        # Check if the character is an upper case letter
        if char.isupper():
            # Add the ASCII code of the character to the sum
            ascii_sum += ord(char)
    
    # Return the sum of ASCII codes
    return ascii_sum

# Test case
input_string = "HelloWorld"
output = method(input_string)
print(f"Input: {input_string}")
print(f"Output: {output}")
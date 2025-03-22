import string

def sum_upper_ascii_codes(input_string):
    """
    This function takes a string as input and returns the sum of the upper characters only' ASCII codes.

    Args:
      input_string: The input string.

    Returns:
      The sum of the upper characters only' ASCII codes.
    """

    # Create an empty list to store the ASCII codes of the upper characters.
    upper_ascii_codes = []

    # Iterate through the input string and check if each character is an uppercase letter.
    for character in input_string:
        if character in string.ascii_uppercase:
            # If the character is an uppercase letter, add its ASCII code to the list.
            upper_ascii_codes.append(ord(character))

    # Return the sum of the ASCII codes in the list.
    return sum(upper_ascii_codes)


# Test case
input_string = "ThisIsAString"
output = sum_upper_ascii_codes(input_string)
print(output)  # Expected output: 768
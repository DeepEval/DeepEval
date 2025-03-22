def method(input_string):
    """
    This function flips the case of a given string. 
    It converts lowercase characters to uppercase and vice versa.

    Args:
        input_string (str): The string to be processed.

    Returns:
        str: The string with its case flipped.
    """
    # Use the swapcase() method to flip the case of the input string
    output = input_string.swapcase()
    
    return output

# Test case for validation
if __name__ == "__main__":
    input_string = "Hello World"
    output = method(input_string)
    print(f"Input: {input_string}")
    print(f"Output: {output}")
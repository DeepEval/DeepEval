def method(input_string):
    """
    This function calculates the number of distinct characters in a given string, 
    regardless of their case.

    Args:
        input_string (str): The string to calculate distinct characters from.

    Returns:
        int: The number of distinct characters in the input string.
    """
    # Convert the string to lower case to handle case sensitivity
    lower_case_string = input_string.lower()
    
    # Use a set to store unique characters
    unique_characters = set(lower_case_string)
    
    # Return the number of unique characters
    return len(unique_characters)

# Test case
input_string = "HelloWorld"
output = method(input_string)
print(f"Distinct characters in '{input_string}': {output}")
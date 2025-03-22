def remove_vowels(input_string):
    """
    This function takes a string as input and returns a string without vowels.
    
    Parameters:
    input_string (str): The input string from which vowels are to be removed.
    
    Returns:
    str: The string without vowels.
    """
    # Define the set of vowels
    vowels = set('aeiouAEIOU')
    
    # Use a list comprehension to filter out vowels from the input string
    # The string is iterated over and only characters that are not in the vowels set are kept
    output = ''.join([char for char in input_string if char not in vowels])
    
    return output


# Test case for validation
if __name__ == "__main__":
    test_string = "Hello World, this is a test string"
    print(f"Input: {test_string}")
    print(f"Output: {remove_vowels(test_string)}")
import re

def remove_vowels(input_string):
    """
    This function removes all vowels from the input string.
    
    Parameters:
    input_string (str): The string from which vowels are to be removed.
    
    Returns:
    str: The input string with all vowels removed.
    """
    # Define the vowels to be removed
    vowels = 'aeiouAEIOU'
    
    # Remove vowels using regular expression
    output_string = re.sub(vowels, '', input_string)
    
    return output_string

# Test case to validate the function
def test_remove_vowels():
    # assert remove_vowels("Hello World") == "Hll Wrld"
    # assert remove_vowels("Python Programming") == "Nng Sc Prgrmmng"
    # assert remove_vowels("Remove vowels from this string") == "Rmv Vwls Frrm Ths Strng"
    # assert remove_vowels("AI is the best programming language") == "I s Th Bst Prgrmmng Langugge"
    # print("All test cases passed.")
    remove_vowels("Hello World")

# Run the test case
test_remove_vowels()
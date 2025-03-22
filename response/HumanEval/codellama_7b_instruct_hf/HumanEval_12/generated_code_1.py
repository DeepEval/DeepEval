import sys

def method(str_list):
    """
    Out of list of strings, return the longest one. Return the first one in case of multiple strings of the same length. Return None in case the input list is empty.
    """
    if not str_list:
        return None
    
    max_length = 0
    longest_string = None
    
    for string in str_list:
        if len(string) > max_length:
            max_length = len(string)
            longest_string = string
        elif len(string) == max_length:
            longest_string = string  # Return the first string in case of tie
    
    return longest_string

# Test Case
str_list = ["hello", "world", "python", "programming"]
longest_string = method(str_list)
print(longest_string)  # Output: programming
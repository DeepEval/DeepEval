def method(s):
    """
    Checks if a given string is 'happy' or not.
    A string is 'happy' if its length is at least 3 and every 3 consecutive letters are distinct.
    
    Parameters:
    s (str): The input string to be checked.
    
    Returns:
    bool: True if the string is 'happy', False otherwise.
    """
    
    # Check if the length of the string is at least 3
    if len(s) < 3:
        return False
    
    # Iterate over the string with a sliding window of size 3
    for i in range(len(s) - 2):
        # Get the current window of 3 characters
        window = s[i:i+3]
        
        # Check if all characters in the window are distinct
        if len(set(window))!= 3:
            return False
    
    # If the loop completes without finding any distinct windows, the string is 'happy'
    return True


# Test case
s = "abcba"
output = method(s)
print(f"Input: {s}, Output: {output}")  # Output: False
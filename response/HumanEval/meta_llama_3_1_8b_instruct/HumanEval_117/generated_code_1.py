import re
from typing import List

def method(s: str, n: int) -> List[str]:
    """
    Returns a list of words from string s that contain exactly n consonants.

    Args:
        s (str): The input string.
        n (int): The number of consonants in each word.

    Returns:
        List[str]: A list of words with exactly n consonants.
    """
    # Define a regular expression pattern to match consonants
    consonant_pattern = r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]'
    
    # Initialize an empty list to store the result
    result = []
    
    # Split the input string into words
    words = s.split()
    
    # Iterate over each word in the list of words
    for word in words:
        # Count the number of consonants in the word
        consonant_count = len(re.findall(consonant_pattern, word))
        
        # Check if the word has exactly n consonants
        if consonant_count == n:
            # If the word has exactly n consonants, add it to the result list
            result.append(word)
    
    # Return the list of words with exactly n consonants
    return result

# Test case
s = "This is a test string with some words that contain 3 consonants each."
n = 3
print(method(s, n))
import re

def count_consonants(s):
    """
    Count the number of consonants in a given string.
    """
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    return sum(1 for c in s if c in consonants)

def method(s, n):
    """
    Return a list of all words from string s that contain exactly n consonants.
    """
    # Split the string into words
    words = re.findall(r'\b\w+\b', s)
    
    # Filter words based on the number of consonants
    filtered_words = [word for word in words if count_consonants(word) == n]
    
    return filtered_words

# Test case
s = "Hello, my friend! Nice to meet you in Cambridge. I am learning Python."
n = 3
print(method(s, n))
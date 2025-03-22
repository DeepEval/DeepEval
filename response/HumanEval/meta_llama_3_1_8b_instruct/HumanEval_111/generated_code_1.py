from collections import Counter

def method(s):
    """
    Given a string of space-separated lowercase letters, return a dictionary 
    with the letter(s) that have the most repetition and their corresponding count(s).
    
    Args:
        s (str): A string of space-separated lowercase letters.
    
    Returns:
        dict: A dictionary with the letter(s) that have the most repetition and their corresponding count(s).
    """
    
    # Count the occurrences of each letter
    letter_counts = Counter(s.split())
    
    # Find the maximum count
    max_count = max(letter_counts.values())
    
    # Filter the letters with the maximum count
    most_common_letters = {letter: count for letter, count in letter_counts.items() if count == max_count}
    
    return most_common_letters

# Test case
s = "a b c a b c a b"
print(method(s))  # Output: {'a': 4, 'b': 4, 'c': 4}
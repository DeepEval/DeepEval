from collections import Counter

def find_max(lst):
    """
    This function accepts a list of strings and returns the word with maximum number of unique characters.
    If multiple strings have maximum number of unique characters, it returns the one which comes first in lexicographical order.

    Parameters:
    lst (list): A list of strings.

    Returns:
    str: The word with maximum number of unique characters.
    """
    max_unique_chars = 0
    max_unique_word = ""

    for word in lst:
        # Count the frequency of each character in the word
        char_count = Counter(word)
        
        # Calculate the number of unique characters in the word
        unique_chars = len(char_count)
        
        # If the word has more unique characters than the current max, update the max
        if unique_chars > max_unique_chars:
            max_unique_chars = unique_chars
            max_unique_word = word
        # If the word has the same number of unique characters as the current max, 
        # update the max if the word comes first in lexicographical order
        elif unique_chars == max_unique_chars and word < max_unique_word:
            max_unique_word = word

    return max_unique_word

# Test case
print(find_max(["name", "of", "string"]))  # Output: "string"
print(find_max(["name", "enam", "game"]))  # Output: "enam"
print(find_max(["aaaaaaa", "bb", "cc"]))  # Output: "aaaaaaa"
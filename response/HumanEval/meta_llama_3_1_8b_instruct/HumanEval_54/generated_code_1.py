def method():
    """
    Checks if two words have the same characters.

    Args:
        word1 (str): The first word.
        word2 (str): The second word.

    Returns:
        bool: True if the words have the same characters, False otherwise.
    """
    def has_same_characters(word1, word2):
        # Convert the words to lowercase and sort the characters
        sorted_word1 = sorted(word1.lower())
        sorted_word2 = sorted(word2.lower())
        
        # Compare the sorted lists of characters
        return sorted_word1 == sorted_word2

    # Test case
    word1 = "Listen"
    word2 = "Silent"
    output = has_same_characters(word1, word2)
    
    return output

# Test case validation
print(method())  # Expected output: True
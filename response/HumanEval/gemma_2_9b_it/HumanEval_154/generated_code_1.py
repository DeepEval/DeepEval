def cycpattern_check(word1, word2):
    """
    Checks if the second word or any of its rotations is a substring in the first word.

    Args:
        word1 (str): The first word.
        word2 (str): The second word.

    Returns:
        bool: True if word2 or any of its rotations is a substring of word1, False otherwise.
    """

    if len(word2) > len(word1):
        return False  

    for i in range(len(word2)):
        rotation = word2[i:] + word2[:i]  
        if rotation in word1:
            return True

    return False

# Test Case
word1 = "himenss"
word2 = "simen"
output = cycpattern_check(word1, word2)
print(f"cycpattern_check('{word1}','{word2}') => {output}")
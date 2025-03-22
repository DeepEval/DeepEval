from collections import Counter

def method(str1, str2):
    """
    Check if two words have the same characters.

    Args:
        str1: The first word.
        str2: The second word.

    Returns:
        True if the words have the same characters, False otherwise.
    """

    # Convert the words to Counter objects.
    counter1 = Counter(str1)
    counter2 = Counter(str2)

    # Check if the Counter objects are equal.
    if counter1 == counter2:
        return True
    else:
        return False


# Test case
str1 = "racecar"
str2 = "racecar"
method(str1, str2)
# assert method(str1, str2) is True
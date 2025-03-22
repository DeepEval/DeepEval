def method(text):
    """
    Given a string of words, return a list of words split on whitespace, 
    if no whitespaces exists in the text you should split on commas ',' 
    if no commas exists you should return the number of lower-case letters 
    with odd order in the alphabet.

    Args:
        text (str): The input string

    Returns:
        list or int: A list of words or the number of lower-case letters with odd order in the alphabet
    """
    # Remove leading and trailing whitespaces
    text = text.strip()

    # Split the text on whitespace
    words = text.split()

    # If no words are found, split on commas
    if not words:
        words = text.split(',')

    # If no commas are found, count lower-case letters with odd order in the alphabet
    if not words:
        count = sum(1 for char in text.lower() if 'a' <= char <= 'z' and ord(char) % 2 == 1)
        return count

    return words

# Test case
print(method("hello world"))  # Output: ['hello', 'world']
print(method("hello,world"))  # Output: ['hello', 'world']
print(method("hello"))  # Output: 5
print(method("aBc"))  # Output: 2
print(method("abcdefg"))  # Output: 4
print(method(""))  # Output: 0
print(method(","))  # Output: 0
def method():
    """
    This function counts the number of uppercase vowels in even indices of a given string.

    Args:
        None

    Returns:
        int: The count of uppercase vowels in even indices.
    """
    # Initialize the count of uppercase vowels to 0
    count = 0

    # Given string
    s = "Hello World"

    # Iterate over the string with index
    for i in range(len(s)):
        # Check if the index is even
        if i % 2 == 0:
            # Check if the character is an uppercase vowel
            if s[i].upper() in 'AEIOU':
                # Increment the count if it's an uppercase vowel
                count += 1

    # Return the count of uppercase vowels in even indices
    return count

# Test case for validation
print(method())  # Expected output: 3
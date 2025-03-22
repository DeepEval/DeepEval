def method(word):
    # Define vowels and consonants
    vowels = "aeiouAEIOU"
    
    # Start from the end of the string
    length = len(word)
    
    # Iterate backwards through the string starting from the second last character
    for i in range(length - 2, 0, -1):  # Start from length - 2 to skip the last character
        # Check if current character is a vowel
        if word[i] in vowels:
            # Check if the characters on both sides are consonants
            if word[i - 1].isalpha() and word[i - 1] not in vowels and word[i + 1].isalpha() and word[i + 1] not in vowels:
                return word[i]  # Return the found vowel
    
    return ""  # Return an empty string if no vowel was found

# Test case
test_word = "bceagd"  # The closest vowel between consonants is 'e'
output = method(test_word)
print(output)  # Expected output: 'e'
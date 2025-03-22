def method(word):
    vowels = 'AEIOUaeiou'
    
    # Start iterating from the second to last character to the second character
    for i in range(len(word) - 2, 0, -1):
        # Check if the current character is a vowel
        if word[i] in vowels:
            # Check if the character before and after the current character are consonants
            if word[i - 1] not in vowels and word[i + 1] not in vowels:
                return word[i]
    
    # Return an empty string if no such vowel is found
    return ""

# Test case
print(method("example"))  # Expected output: "a"
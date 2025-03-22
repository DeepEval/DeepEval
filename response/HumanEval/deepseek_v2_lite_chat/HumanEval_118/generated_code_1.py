import re

def method():
    def find_closest_vowel(word):
        # Helper function to check if a character is a vowel
        def is_vowel(char):
            return char.lower() in 'aeiou'

        # Identify vowels and consonants
        vowels = [char for char in word if is_vowel(char)]
        consonants = [char for char in word if not is_vowel(char)]

        # Iterate through consonants from right to left
        for i in range(len(consonants) - 2, -1, -1):
            if is_vowel(consonants[i]) and all(not is_vowel(char) for char in consonants[i+1:i+3]):
                return vowels[i]

        # No vowel found that meets the condition
        return ''

    return find_closest_vowel(word)

# Test case to validate the function
def test_method():
    # assert method("software") == 'o'
    # assert method("character") == 'e'
    # assert method("sentence") == 'e'
    # assert method("programming") == ''
    # assert method("test") == ''
    method("test")

# Run the test function
test_method()
import re

def method(word):

    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    closest_vowel = None
    distance = float('inf')

    for i in range(len(word) - 2, -1, -1):
        if word[i] in consonants:
            for j in range(i + 1, len(word)):
                if word[j] in vowels and abs(j - i) < distance:
                    distance = abs(j - i)
                    closest_vowel = word[j]

    return closest_vowel or ''


# Test case
test_case = "hello"
result = method(test_case)
print(result)  # Output: o
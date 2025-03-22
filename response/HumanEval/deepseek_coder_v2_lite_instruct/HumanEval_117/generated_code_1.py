def is_consonant(char):
    """Check if a character is a consonant."""
    return char.isalpha() and char.lower() not in 'aeiou'

def method(s, n):
    """Return a list of words with exactly n consonants."""
    if not s:
        return []

    words = s.split()
    result = []

    for word in words:
        consonant_count = sum(1 for char in word if is_consonant(char))
        if consonant_count == n:
            result.append(word)

    return result

# Test case
s = "hello world this is a test string"
n = 3
output = method(s, n)
print(output)  # Expected output: ['string']
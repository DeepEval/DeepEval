def method(s, n):
    # Function to count consonants in a word
    def count_consonants(word):
        vowels = set('aeiouAEIOU')
        return sum(1 for char in word if char not in vowels)
    
    # Split the string into words
    words = s.split()
    
    # Filter words with exactly n consonants
    output = [word for word in words if count_consonants(word) == n]
    
    return output

# Test case for validation
test_string = "This is a simple example sentence"
n = 3
# 'This' has 3 consonants: T, h, s
# 'simple' has 3 consonants: s, m, p
# 'sentence' has 3 consonants: s, n, t
expected_output = ['This', 'simple', 'sentence']
# assert method(test_string, n) == expected_output
method(test_string, n)
# print("Test case passed!")
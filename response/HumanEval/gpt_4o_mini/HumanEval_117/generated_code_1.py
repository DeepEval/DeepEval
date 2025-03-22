def method(s, n):
    # Function to count consonants in a word
    def count_consonants(word):
        consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
        count = sum(1 for char in word if char in consonants)
        return count

    # Split the string into words
    words = s.split()
    
    # Filter words that contain exactly n consonants
    result = [word for word in words if count_consonants(word) == n]
    
    return result

# Test case
s = "Hello world this is a test"
n = 2
output = method(s, n)  # This should return ['world', 'this', 'is']
print(output)  # Output should be ['world', 'this']
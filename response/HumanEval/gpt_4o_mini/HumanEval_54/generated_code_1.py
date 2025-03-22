from collections import Counter

def method(word1, word2):
    # Count the characters in both words
    count1 = Counter(word1)
    count2 = Counter(word2)

    # Compare the two counters
    output = count1 == count2
    return output

# Test case
word1 = "listen"
word2 = "silent"
result = method(word1, word2)
print(f"Do '{word1}' and '{word2}' have the same characters? {result}")
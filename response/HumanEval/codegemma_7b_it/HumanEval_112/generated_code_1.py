from collections import Counter
import re

def method():
    s = "ababa"
    c = "ba"

    # Count the occurrences of each character in c
    c_counts = Counter(c)

    # Remove characters in s that are present in c
    s = re.sub(r'|'.join(c_counts.keys()), '', s)

    # Check if the remaining string is palindrome
    is_palindrome = s == s[::-1]

    return s, is_palindrome


# Test case
s = "ababa"
c = "ba"
output = method()
print(output)  # Output: ('a', True)
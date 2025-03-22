def method(strings):
    # Filter out strings with odd lengths
    even_length_strings = [s for s in strings if len(s) % 2 == 0]
    
    # Sort the list based on the length and then alphabetically
    sorted_strings = sorted(even_length_strings, key=lambda x: (len(x), x))
    
    return sorted_strings

# Test case
strings = ["apple", "banana", "cat", "dog", "elephant", "fox", "giraffe"]
output = method(strings)
print(output)
def method(strings):
    # Filter out strings with odd lengths
    filtered_strings = [s for s in strings if len(s) % 2 == 0]
    
    # Sort the list by length and then alphabetically
    sorted_strings = sorted(filtered_strings, key=lambda s: (len(s), s))
    
    return sorted_strings

# Test case for validation
if __name__ == "__main__":
    test_input = ["apple", "banana", "pear", "kiwi", "grape", "melon", "fig", "date", "cherry"]
    output = method(test_input)
    print(output)  # Expected output: ['date', 'kiwi', 'melon', 'banana', 'cherry']
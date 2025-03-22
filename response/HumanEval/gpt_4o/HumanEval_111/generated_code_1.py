def method(input_string):
    # Dictionary to store the frequency of each letter
    letter_count = {}
    
    # Count each letter in the string
    for letter in input_string.split():
        letter_count[letter] = letter_count.get(letter, 0) + 1

    # Find the maximum occurrence
    max_count = max(letter_count.values())
    
    # Find all letters with the maximum occurrence
    output = {letter: count for letter, count in letter_count.items() if count == max_count}

    return output

# Test case
test_string = "a a b b c c c d e"
print(method(test_string))
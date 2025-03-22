def method(input_string):
    from collections import Counter

    # Split the input string into individual letters
    letters = input_string.split()

    # Count the occurrences of each letter
    letter_counts = Counter(letters)

    # Find the maximum occurrence count
    max_count = max(letter_counts.values())

    # Create a dictionary for letters with the maximum count
    output = {letter: count for letter, count in letter_counts.items() if count == max_count}

    return output

# Test case
input_string = "a b c a b a d c b"
output = method(input_string)
print(output)  # Expected output: {'a': 3, 'b': 3}
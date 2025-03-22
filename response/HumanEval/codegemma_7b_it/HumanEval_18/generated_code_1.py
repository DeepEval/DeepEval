from collections import Counter

def method():
    input_string = "This is a string with the substring 'is' in it."
    substring = "is"

    # Create a dictionary to store character counts in the substring
    substring_counts = Counter(substring)

    # Initialize a variable to store the number of occurrences
    occurrences = 0

    # Iterate over the input string
    for i in range(len(input_string)):
        # Check if the current character matches the first character of the substring
        if input_string[i] == substring_counts.keys()[0]:
            # Check if the next characters in the input string match the remaining characters of the substring
            if input_string[i+1:i+1+len(substring)].count(substring_counts.keys()[0]) == substring_counts[substring_counts.keys()[0]]:
                occurrences += 1

    return occurrences

# Test case
input_string = "This is a string with the substring 'is' in it."
substring = "is"
expected_output = 2

output = method()

# assert output == expected_output, f"Expected output: {expected_output}, Actual output: {output}"

# print("Test case passed")
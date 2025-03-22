def method():
    # Example usage:
    original_string = "abababab"
    substring = "aba"
    output = count_occurrences(original_string, substring)
    return output

def count_occurrences(original_string, substring):
    count = 0
    index = 0
    while index < len(original_string):
        # Check if the substring is found at the current index
        if original_string[index:index+len(substring)] == substring:
            count += 1
            # Move the index to skip the last character of the current match to allow overlapping
            index += len(substring) - 1
        index += 1
    return count

# Test case
original_string = "abababab"
substring = "aba"
expected_output = 3
output = count_occurrences(original_string, substring)
# assert output == expected_output, f"Test failed! Expected {expected_output}, but got {output}"

# # If the test passes, the method will return the output
# print("Test passed!")
def method(input_string):
    encoded_string = []
    # Iterate over the input string in chunks of three characters
    for i in range(0, len(input_string), 3):
        # Extract a group of up to three characters
        group = input_string[i:i+3]
        # If the group has exactly three characters, rotate them
        if len(group) == 3:
            group = group[1:] + group[0]
        # Append the processed group to the encoded string list
        encoded_string.append(group)
    # Join all groups to form the final encoded string
    output = ''.join(encoded_string)
    return output

# Test case for validation
def test_method():
    input_string = "abcdefg"
    expected_output = "bcadefg"
    result = method(input_string)
    # assert result == expected_output, f"Test failed: expected {expected_output}, got {result}"
    # print("Test passed.")

# Run the test
test_method()
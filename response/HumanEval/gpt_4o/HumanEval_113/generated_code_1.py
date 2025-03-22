def method(input_list):
    # Helper function to determine if a character is an odd digit
    def is_odd_digit(char):
        # Check if the character is a digit and is odd
        return char in '13579'

    output = []
    
    for string in input_list:
        # Count the number of odd digits in the current string
        odd_count = sum(1 for char in string if is_odd_digit(char))
        output.append(odd_count)
    
    return output

# Test case for validation
test_input = ["123", "4567", "89", "02468", "13579"]
expected_output = [2, 2, 1, 0, 5]
assert method(test_input) == expected_output

print("Test passed.")
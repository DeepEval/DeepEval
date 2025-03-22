def method():
    import re

    # Function to count odd digits in a string
    def count_odd_digits(s):
        return sum(int(digit) % 2 for digit in s)

    # Input list of strings
    input_list = ["123", "456", "789", "abc", "def"]

    # List to store the count of odd digits for each string
    output = []

    # Iterate through each string in the input list
    for s in input_list:
        # Count the number of odd digits in the current string
        odd_digits_count = count_odd_digits(s)
        # Append the count to the output list
        output.append(odd_digits_count)

    return output

# Test case to validate the function
test_case = [1, 2, 0, 1, 1]  # Expected output for the test case
# assert method() == test_case, "Test case validation failed"

# Print the output for validation
print("Output:", method())
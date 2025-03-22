def method(brackets):
    # Initialize a counter for open brackets
    open_count = 0

    # Iterate through each character in the string
    for char in brackets:
        if char == '<':
            open_count += 1  # Increment for an opening bracket
        elif char == '>':
            open_count -= 1  # Decrement for a closing bracket

        # If the count goes negative, we have more closing than opening brackets
        if open_count < 0:
            return False

    # Return True if all opening brackets are matched, False otherwise
    return open_count == 0

# Test case
test_case = "<<>>"
output = method(test_case)
print(output)  # Expected output: True
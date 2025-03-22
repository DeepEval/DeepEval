def method(strings):
    # Check if the input list is empty
    if not strings:
        return None

    # Initialize variables to track the longest string
    longest_string = ""
    
    # Iterate through the list of strings
    for string in strings:
        # Update longest_string if current string is longer
        if len(string) > len(longest_string):
            longest_string = string
            
    # Return the longest string found
    return longest_string

# Test case for validation
test_case = ["apple", "banana", "cherry", "kiwi", "mango"]
output = method(test_case)
print(output)  # Expected output: "banana" (the longest string)
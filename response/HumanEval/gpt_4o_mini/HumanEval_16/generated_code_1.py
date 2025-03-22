def method(input_string):
    # Convert the string to lowercase to ignore case
    lower_case_string = input_string.lower()
    
    # Use a set to find distinct characters
    distinct_characters = set(lower_case_string)
    
    # Filter out non-alphabetic characters if needed
    distinct_characters = {char for char in distinct_characters if char.isalpha()}
    
    # Count the number of distinct characters
    output = len(distinct_characters)
    
    return output

# Test case
test_string = "Hello, World!"
result = method(test_string)
print(f"Distinct characters in '{test_string}': {result}")
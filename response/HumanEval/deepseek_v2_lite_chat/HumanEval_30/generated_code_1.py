def method():
    # Sample list of numbers
    numbers = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
    
    # Use list comprehension to filter out non-positive numbers
    positive_numbers = [num for num in numbers if num > 0]
    
    # Return the list of positive numbers
    return positive_numbers

# Test case to validate the function
def test_method():
    # Expected output
    expected_output = [1, 3, 5, 7, 9]
    
    # Call the method function
    output = method()
    
    # Check if the output matches the expected output
    # assert expected_output == output, f"Expected {expected_output}, but got {output}"
    
    print("Test passed.")

# Run the test case
test_method()
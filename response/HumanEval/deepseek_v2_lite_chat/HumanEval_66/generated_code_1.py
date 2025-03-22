import string

def method(input_string):
    # Initialize the sum to zero
    sum_upper_ascii = 0
    
    # Loop through each character in the input string
    for char in input_string:
        # Check if the character is an uppercase letter
        if char.isupper():
            # Add the ASCII value of the uppercase letter to the sum
            sum_upper_ascii += ord(char)
    
    # Return the sum of the uppercase letter ASCII codes
    return sum_upper_ascii

# Test case to validate the function
def test_method():
    # Test with a string containing uppercase letters
    test_string = "HELLO"
    expected_output = 72  # ASCII values for 'H', 'E', 'L', 'L', 'O'
    
    # Call the method and store the result
    output = method(test_string)
    
    # Compare the output with the expected result
    if output == expected_output:
        print("Test passed!")
    else:
        print("Test failed!")
        print("Expected output:", expected_output)
        print("Actual output:", output)

# Run the test function
test_method()
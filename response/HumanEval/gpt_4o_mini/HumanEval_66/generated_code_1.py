def method(input_string):
    # Initialize a variable to hold the sum of ASCII values
    ascii_sum = 0
    
    # Iterate through each character in the input string
    for char in input_string:
        # Check if the character is an uppercase letter
        if char.isupper():
            # Add the ASCII value of the uppercase letter to the sum
            ascii_sum += ord(char)
    
    # Return the total sum of ASCII values of uppercase letters
    return ascii_sum

# Test case for validation
if __name__ == "__main__":
    test_string = "Hello World! This is a Test String with Uppercase Letters."
    output = method(test_string)
    print(f"Sum of ASCII values of uppercase characters: {output}")
def method(input_string):
    # Define a helper function to shift a character by 5
    def shift_char(char):
        # Calculate the ASCII value of the character
        ascii_val = ord(char)
        # Shift the character by 5
        new_ascii_val = ascii_val + 5
        # If the new value exceeds 'z', subtract 26 to wrap around to the beginning of the alphabet
        if char.isupper():
            new_char = chr(new_ascii_val % 26 + ord('A'))
        else:
            new_char = chr(new_ascii_val % 26 + ord('a'))
        return new_char
    
    # Encode the input string
    encoded_string = ''.join(shift_char(char) for char in input_string)
    
    return encoded_string

# Example test case
test_input = "Hello, World!"
test_output = "Mjqqt, Btwqi!"
method(test_input)
# assert method(test_input) == test_output
# print("All test cases pass.")
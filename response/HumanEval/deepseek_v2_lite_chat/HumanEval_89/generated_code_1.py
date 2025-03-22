import string

def encrypt(input_str):
    # Define the alphabet
    alphabet = string.ascii_lowercase
    
    # Function to shift characters in the alphabet
    def shift_char(c, shift=2):
        # Check if the character is a letter
        if c.isalpha():
            # Find the position of the character in the alphabet
            pos = alphabet.index(c.lower())
            # Calculate the new position after shifting
            new_pos = (pos + shift * 2) % 26
            # Return the new character or the original character if it's not a letter
            return alphabet[new_pos] if c.isalpha() else c
        else:
            return c
    
    # Encrypt the input string
    encrypted_str = ''.join(shift_char(c) for c in input_str)
    
    return encrypted_str

# Test case
def test_encrypt():
    # assert encrypt("hello") == "jgnnq"
    # assert encrypt("python") == "srhrv"
    # assert encrypt("encrypt") == "jntbaf"
    # print("All test cases passed")
    print(encrypt("hello"))

# Run the test case
test_encrypt()
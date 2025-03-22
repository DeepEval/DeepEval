def encrypt(input_string):
    # Define the alphabet
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    # Create a mapping for the rotated alphabet
    rotated_alphabet = alphabet[2:] + alphabet[:2]
    
    # Create a translation table for the alphabet
    translation_table = str.maketrans(alphabet + alphabet.upper(), rotated_alphabet + rotated_alphabet.upper())
    
    # Encrypt the input string
    encrypted_string = input_string.translate(translation_table)
    
    return encrypted_string

def method():
    # Test case
    input_string = "Hello, World!"
    output = encrypt(input_string)
    return output

# Run the method to get the output
output = method()
print(output)
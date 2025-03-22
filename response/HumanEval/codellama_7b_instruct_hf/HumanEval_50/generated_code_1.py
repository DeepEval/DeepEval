import string
def method():
    # Get the input string from the user
    # input_string = input("Enter a string: ")

    # 修改
    input_string = "Hello, world!"

    # Create an empty list to store the encoded characters
    encoded_characters = []

    # Iterate through the input string
    for character in input_string:
        # Check if the character is a letter
        if character.isalpha():
            # Shift the character by 5 in the alphabet
            if character.islower():
                encoded_character = chr(((ord(character) - ord('a') + 5) % 26) + ord('a'))
            else:
                encoded_character = chr(((ord(character) - ord('A') + 5) % 26) + ord('A'))
            encoded_characters.append(encoded_character)
        else:
            # Keep non-letter characters unchanged
            encoded_characters.append(character)

    # Join the encoded characters into a string
    encoded_string = ''.join(encoded_characters)

    # Return the encoded string
    return encoded_string

# Test case
test_case = "Hello, world!"
encoded_test_case = method()
print(f"Test case: {test_case}")
print(f"Encoded test case: {encoded_test_case}")
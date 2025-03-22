def method(input_string):
    encoded_string = ""
    for char in input_string:
        if char.isalpha():
            if char.islower():
                encoded_char = chr((ord(char) - ord('a') + 5) % 26 + ord('a'))
            else:
                encoded_char = chr((ord(char) - ord('A') + 5) % 26 + ord('A'))
            encoded_string += encoded_char
        else:
            encoded_string += char
    return encoded_string

# Test case
input_string = "Hello, World!"
output = method(input_string)
print(f"Input: {input_string}")
print(f"Output: {output}")
def method(a, b):
    # Initialize an empty list to store the result of XOR operation
    result = []

    # Iterate over the characters in the strings a and b
    for char_a, char_b in zip(a, b):
        # Perform XOR operation on the characters and append the result to the list
        # XOR of '1' and '0' is '1', and XOR of '0' and '1' is '1'
        # XOR of '0' and '0' is '0', and XOR of '1' and '1' is '0'
        result.append(str(int(char_a) ^ int(char_b)))

    # Join the list into a string and return it
    output = ''.join(result)
    return output

# Test case for validation
a = "1101"
b = "1011"
print(method(a, b))  # Expected output: "0110"
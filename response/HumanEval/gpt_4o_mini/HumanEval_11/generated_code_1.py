def method(a, b):
    # Ensure both strings are of the same length
    if len(a) != len(b):
        raise ValueError("Strings must be of the same length")
    
    # Perform binary XOR
    result = []
    for char_a, char_b in zip(a, b):
        # XOR operation: '0' with '0' -> '0', '1' with '1' -> '0', '0' with '1' -> '1', '1' with '0' -> '1'
        result.append('1' if char_a != char_b else '0')
    
    # Join the list into a string
    output = ''.join(result)
    return output

# Test case for validation
a = "1101"
b = "1011"
output = method(a, b)
print(output)  # Expected output: "0110"
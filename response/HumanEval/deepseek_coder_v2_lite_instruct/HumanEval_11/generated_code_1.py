def method():
    def binary_xor(a, b):
        if len(a) != len(b):
            raise ValueError("Input strings must be of the same length")
        
        result = []
        for bit_a, bit_b in zip(a, b):
            if bit_a == bit_b:
                result.append('0')
            else:
                result.append('1')
        
        return ''.join(result)

    # Example usage:
    a = "1100"
    b = "1010"
    output = binary_xor(a, b)
    return output

# # Test case
# def test_binary_xor():
#     a = "1100"
#     b = "1010"
#     expected_output = "0110"
#     assert binary_xor(a, b) == expected_output, f"Test failed: binary_xor({a}, {b}) should be {expected_output}"

#     a = "1111"
#     b = "0000"
#     expected_output = "1111"
#     assert binary_xor(a, b) == expected_output, f"Test failed: binary_xor({a}, {b}) should be {expected_output}"

#     a = "1001"
#     b = "1001"
#     expected_output = "0000"
#     assert binary_xor(a, b) == expected_output, f"Test failed: binary_xor({a}, {b}) should be {expected_output}"

#     print("All tests passed!")

# # Run the test case
# test_binary_xor()

print(method())
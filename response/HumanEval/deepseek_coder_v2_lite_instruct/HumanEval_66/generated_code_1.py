def method():
    import string

    def sum_of_upper_ascii(input_string):
        return sum(ord(char) for char in input_string if char in string.ascii_uppercase)

    # Example usage
    input_string = "Hello World! 123"
    output = sum_of_upper_ascii(input_string)
    return output

# # Test case
# def test_sum_of_upper_ascii():
#     assert method() == 307  # Sum of ASCII codes of uppercase characters in "Hello World! 123" is 72 + 79 + 87 + 84 + 82 + 69 = 307
#     print("Test passed!")

# # Run the test
# test_sum_of_upper_ascii()

method()
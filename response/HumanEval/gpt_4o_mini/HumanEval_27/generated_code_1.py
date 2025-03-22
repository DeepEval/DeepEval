def method(input_string):
    # Flip the case of the characters in the input string
    output = input_string.swapcase()
    return output

# Test case for validation
if __name__ == "__main__":
    test_string = "Hello, World!"
    result = method(test_string)
    print("Input String: ", test_string)
    print("Flipped String: ", result)
def method(input_string):
    # Return the length of the given string
    output = len(input_string)
    return output

# Test case
if __name__ == "__main__":
    test_string = "Hello, world!"
    result = method(test_string)
    print(f"The length of the string '{test_string}' is: {result}")
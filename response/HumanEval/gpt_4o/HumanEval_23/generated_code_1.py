def method(input_string):
    # Calculate the length of the input string
    output = len(input_string)
    return output

# Test case for validation
if __name__ == "__main__":
    test_string = "Hello, World!"
    print(f"The length of the string '{test_string}' is: {method(test_string)}")
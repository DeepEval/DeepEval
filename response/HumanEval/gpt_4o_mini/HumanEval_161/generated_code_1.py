def method(s):
    if any(char.isalpha() for char in s):  # Check if there are any letters in the string
        output = ''.join(char.swapcase() if char.isalpha() else char for char in s)
    else:
        output = s[::-1]  # Reverse the string if there are no letters

    return output

# Test case
test_string = "Hello, World!"
result = method(test_string)
print(result)  # Expected output: "hELLO, wORLD!"

# Another test case with no letters
test_string_no_letters = "12345!@#"
result_no_letters = method(test_string_no_letters)
print(result_no_letters)  # Expected output: "#@!54321"
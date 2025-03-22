def method():
    input_string = "This is a test string."
    vowels = "aeiouAEIOU"
    output = "".join([char for char in input_string if char not in vowels])
    return output

# Test Case
result = method()
print(f"Input: {input_string}")
print(f"Output: {result}")
def method(input_string):
    vowels = "aeiouAEIOU"
    output = ''.join([char for char in input_string if char not in vowels])
    return output

# Test case for validation
def test_remove_vowels():
    test_string = "Hello, World!"
    expected_output = "Hll, Wrld!"
    result = method(test_string)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")

# Run the test case
test_remove_vowels()
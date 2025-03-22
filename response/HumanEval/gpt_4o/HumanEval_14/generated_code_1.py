def method(input_string):
    prefixes = []
    for i in range(1, len(input_string) + 1):
        prefixes.append(input_string[:i])
    return prefixes

# Test case for validation
def test_method():
    input_str = "hello"
    expected_output = ["h", "he", "hel", "hell", "hello"]
    result = method(input_str)
    assert result == expected_output, f"Test failed: expected {expected_output}, got {result}"
    print("Test passed.")

# Run the test case
test_method()
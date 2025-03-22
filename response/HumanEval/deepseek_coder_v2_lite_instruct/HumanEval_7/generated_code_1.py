def method(input_list, substring):
    output = [s for s in input_list if substring in s]
    return output

# Test case
def test_method():
    input_list = ["apple", "banana", "cherry", "date", "fig", "grape"]
    substring = "a"
    expected_output = ["apple", "banana", "date", "grape"]
    result = method(input_list, substring)
    # assert result == expected_output, f"Test failed: expected {expected_output}, but got {result}"
    # print("Test passed!")

# Running the test case
test_method()
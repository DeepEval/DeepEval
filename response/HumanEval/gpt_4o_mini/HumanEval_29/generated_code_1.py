def method(input_list, prefix):
    # Use list comprehension to filter strings that start with the given prefix
    output = [s for s in input_list if s.startswith(prefix)]
    return output

# Test case for validation
if __name__ == "__main__":
    test_list = ["apple", "banana", "apricot", "blueberry", "avocado"]
    test_prefix = "ap"
    expected_output = ["apple", "apricot", "avocado"]
    
    result = method(test_list, test_prefix)
    # print("Filtered output:", result)  # Should print: Filtered output: ['apple', 'apricot', 'avocado']
    # assert result == expected_output, "Test case failed!"
    # print("Test case passed!")
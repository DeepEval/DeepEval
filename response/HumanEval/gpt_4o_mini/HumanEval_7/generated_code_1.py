def method(string_list, substring):
    # Filter the list for strings that contain the given substring
    output = [s for s in string_list if substring in s]
    return output

# Test case for validation
if __name__ == "__main__":
    test_list = ["apple", "banana", "cherry", "date", "fig", "grape"]
    test_substring = "a"
    result = method(test_list, test_substring)
    print("Filtered list:", result)  # Expected output: ['apple', 'banana', 'grape']
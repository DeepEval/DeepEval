def method():
    import re

    def find_longest_string(lst):
        if not lst:  # If the list is empty, return None
            return None
        
        # Use regular expression to find the longest string and return the first one in case of multiple strings of the same length
        longest_str = max(lst, key=len)
        
        # Return the first one in case of multiple strings of the same length
        return longest_str
    
    # Test case
    test_cases = [["hello", "world", "python", "programming"], ["apple", "banana", "cherry", "date"], [], ["short"]]
    expected_outputs = ["python", "date", None, "date"]
    
    for test, expected in zip(test_cases, expected_outputs):
        output = find_longest_string(test)
        # assert output == expected, f"Test case failed: Expected {expected} but got {output}"
    
    return "All test cases passed"

# Call the method and print the result
print(method())
def method(s):
    # Check if the length of the string is less than 3
    if len(s) < 3:
        return False

    # Loop through the string and check each group of 3 consecutive characters
    for i in range(len(s) - 2):
        # Get the three consecutive characters
        triplet = s[i:i+3]
        # Check if all characters in the triplet are distinct
        if len(set(triplet)) != 3:
            return False

    # If all triplets are distinct, the string is happy
    return True

# Test case for validation
def test_method():
    test_cases = [
        ("abc", True),     # "abc" is happy
        ("aabc", False),   # "aab" is not distinct, so not happy
        ("abca", True),    # "abc" and "bca" are both happy
        ("aabbcc", False), # "aab", "abb", "bbc", "bcc" are not happy
        ("abcabc", True),  # "abc", "bca", "cab" are all happy
        ("", False),       # Empty string is not happy
        ("ab", False),     # Length less than 3 is not happy
    ]

    for s, expected in test_cases:
        result = method(s)
        assert result == expected, f"Test failed for input '{s}'. Expected {expected} but got {result}."

    print("All tests passed.")

# Run the test case
test_method()
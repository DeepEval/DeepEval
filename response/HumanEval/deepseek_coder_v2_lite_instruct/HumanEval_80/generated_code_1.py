def method():
    s = "example"  # Example input
    output = is_happy(s)
    return output

def is_happy(s):
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i + 1] or s[i] == s[i + 2] or s[i + 1] == s[i + 2]:
            return False
    return True

# Test case
test_case_string = "abcdef"
print(f"Test case: {test_case_string}")
print(f"Is '{test_case_string}' a happy string? {method()}")
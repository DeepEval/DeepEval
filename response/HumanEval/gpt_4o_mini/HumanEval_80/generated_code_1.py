def method(s):
    # Check if the length of the string is at least 3
    if len(s) < 3:
        return False
    
    # Iterate over the string, checking each triplet
    for i in range(len(s) - 2):
        # Get the current triplet
        triplet = s[i:i+3]
        # Check if all characters in the triplet are distinct
        if len(set(triplet)) != 3:
            return False
            
    return True

# Test case
test_string = "abcde"  # This should return True
output = method(test_string)
print(f"The string '{test_string}' is happy: {output}")

test_string = "aabbc"  # This should return False
output = method(test_string)
print(f"The string '{test_string}' is happy: {output}")

test_string = "aaabbb"  # This should return False
output = method(test_string)
print(f"The string '{test_string}' is happy: {output}")

test_string = "xyz"  # This should return True
output = method(test_string)
print(f"The string '{test_string}' is happy: {output}")
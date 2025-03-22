def method(s):
    # Define the uppercase vowels
    uppercase_vowels = {'A', 'E', 'I', 'O', 'U'}
    
    # Initialize the count of uppercase vowels
    count = 0
    
    # Iterate over the string checking only even indices
    for i in range(0, len(s), 2):  # Step by 2 to get even indices
        if s[i] in uppercase_vowels:
            count += 1
    
    # Return the count
    return count

# Test case for validation
test_string = "AbEcIdOfUg"  # A and I are uppercase vowels at even indices
print(method(test_string))  # Expected output: 2
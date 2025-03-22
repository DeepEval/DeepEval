def method(s):
    # Define the set of uppercase vowels
    uppercase_vowels = {'A', 'E', 'I', 'O', 'U'}
    
    # Initialize a counter for uppercase vowels at even indices
    count = 0
    
    # Iterate through the string, checking only even indices
    for i in range(0, len(s), 2):
        if s[i] in uppercase_vowels:
            count += 1
            
    return count

# Test case for validation
if __name__ == "__main__":
    test_string = "AbEcIdOfU"
    output = method(test_string)
    print("Number of uppercase vowels at even indices:", output)  # Expected output: 4
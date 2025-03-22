def method():
    # Define the string s
    s = "HelloWorld"
    
    # Define the set of uppercase vowels
    uppercase_vowels = {'A', 'E', 'I', 'O', 'U'}
    
    # Initialize the count of uppercase vowels at even indices
    count = 0
    
    # Iterate over the string, checking only even indices
    for i in range(0, len(s), 2):
        if s[i] in uppercase_vowels:
            count += 1
    
    # Return the final count
    output = count
    return output

# Test case
print(method())  # Expected output: 1 (since 'H' is the only uppercase vowel at an even index)
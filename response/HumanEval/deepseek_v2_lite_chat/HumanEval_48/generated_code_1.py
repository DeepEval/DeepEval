def method(input_string):
    # Remove all non-alphanumeric characters (spaces, punctuation, etc.) and convert to lowercase
    cleaned_string = ''.join(filter(str.isalnum, input_string)).lower()
    
    # Compare the cleaned string with its reverse
    if cleaned_string == cleaned_string[::-1]:
        output = True
    else:
        output = False
    
    return output

# Test case to validate the function
test_string = "A man, a plan, a canal, Panama"
print(method(test_string))  # Expected output: True
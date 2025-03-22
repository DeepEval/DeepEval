def method(input_string):
    # Import necessary packages
    import re

    # Define sentence delimiters
    delimiters = r'[.!?]'
    
    # Split the input string into sentences
    sentences = re.split(delimiters, input_string)
    
    # Initialize boredom count
    boredom_count = 0
    
    # Iterate through the sentences and count those starting with "I"
    for sentence in sentences:
        stripped_sentence = sentence.strip()  # Remove leading and trailing whitespace
        if stripped_sentence.startswith("I"):
            boredom_count += 1

    # Return the final count of boredoms
    return boredom_count

# Test case for validation
test_input = "I am bored! How about you? I think I need a break. Let's go!"
output = method(test_input)
print(output)  # Expected output: 2
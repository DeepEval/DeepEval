def method(text):
    # Split the text into sentences using '.', '?', and '!' as delimiters
    import re
    sentences = re.split(r'[.!?]', text)
    
    # Initialize boredom count
    boredom_count = 0
    
    # Iterate over each sentence to check if it starts with "I"
    for sentence in sentences:
        # Strip any leading or trailing spaces
        trimmed_sentence = sentence.strip()
        
        # Check if the sentence starts with "I"
        if trimmed_sentence.startswith("I "):
            boredom_count += 1
    
    # Return the final count of boredom sentences
    return boredom_count

# Test case for validation
text = "I went to the store. Do you want anything? I bought apples! I love apples."
print(method(text))  # Expected output: 3
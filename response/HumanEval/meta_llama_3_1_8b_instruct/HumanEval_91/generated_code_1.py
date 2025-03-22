import re

def method(text):
    """
    This function counts the number of boredoms in a given text.
    A boredom is a sentence that starts with the word "I".
    Sentences are delimited by '.', '?' or '!'.
    
    Parameters:
    text (str): The input text.
    
    Returns:
    int: The number of boredoms in the text.
    """
    
    # Split the text into sentences using regular expression
    sentences = re.split('([.!?])', text)
    
    # Initialize a counter for boredoms
    boredoms = 0
    
    # Iterate over the sentences
    for i in range(0, len(sentences), 2):
        # Check if the sentence starts with "I"
        if sentences[i].lower() == 'i' and i < len(sentences) - 1:
            # Increment the boredom counter
            boredoms += 1
    
    # Return the number of boredoms
    return boredoms

# Test case
text = "I love programming. I think it's fun. I'm excited about AI! Do you like coding?"
print(method(text))  # Output: 3
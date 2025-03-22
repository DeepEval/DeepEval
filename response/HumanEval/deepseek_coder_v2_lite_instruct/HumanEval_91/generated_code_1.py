def method():
    # Sample input string
    input_string = "I am happy! Are you? I was surprised. This is amazing! I feel great."

    # Split the input string into sentences based on delimiters '.', '?', '!'
    import re
    sentences = re.split(r'[.!?]', input_string)

    # Initialize the count of boredoms
    boredom_count = 0

    # Iterate through each sentence
    for sentence in sentences:
        # Strip leading and trailing whitespace from the sentence
        sentence = sentence.strip()
        # Check if the sentence starts with 'I'
        if sentence.startswith('I'):
            boredom_count += 1

    # Prepare the output
    output = boredom_count

    return output

# Test case
def test_method():
    # assert method() == 3, "Test case failed: Expected 3 boredoms, but got something else."
    # print("Test case passed.")
    method()

# Run the test case
test_method()
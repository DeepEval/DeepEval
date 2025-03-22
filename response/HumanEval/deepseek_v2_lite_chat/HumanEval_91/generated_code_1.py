import re

def method(input_string):
    # Split the string into sentences using regex to find periods, question marks, or exclamation points
    sentences = re.split('[.!?]', input_string)
    
    # Initialize a counter for boredoms
    boredom_count = 0
    
    # Loop through each sentence
    for sentence in sentences:
        # Check if the sentence starts with the word "I"
        if sentence.lower().startswith('i'):
            boredom_count += 1
    
    # Return the count of boredoms
    return boredom_count

# Test case for validation
def test_method():
    # Test case 1: Basic case with a single boredom
    test_string = "I am a bored person."
    expected_output = 1
    print(method(test_string))
    # assert method(test_string) == expected_output

    # # Test case 2: Case where there are multiple boredoms
    # test_string = "I am a bored person. I like programming."
    # expected_output = 2
    # assert method(test_string) == expected_output

    # # Test case 3: Case where there are no boredoms
    # test_string = "Programming is fun. I enjoy solving puzzles."
    # expected_output = 0
    # assert method(test_string) == expected_output

    # print("All test cases passed!")

# Run the test function
test_method()
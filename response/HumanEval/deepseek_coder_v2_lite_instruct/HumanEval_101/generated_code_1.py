import re

def method():
    # Example input string
    input_string = "Hello, world! This is a test, and I'm learning Python."
    
    # Split the string into words using regular expression
    # The pattern [ ,]+ matches one or more spaces or commas
    words = re.split('[ ,]+', input_string)
    
    # Filter out any empty strings that might result from consecutive delimiters
    words = [word for word in words if word]
    
    # Return the final output
    output = words
    return output

# # Test case
# def test_method():
#     assert method() == ['Hello', 'world', 'This', 'is', 'a', 'test', "and", "I'm", 'learning', 'Python']
#     print("Test passed!")

# # Run the test
# test_method()

print(method())
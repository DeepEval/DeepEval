def method(input_string):
    def sort_word(word):
        return ''.join(sorted(word))
    
    words = input_string.split(' ')
    sorted_words = [sort_word(word) for word in words]
    output = ' '.join(sorted_words)
    
    return output

# Test case
input_string = "hello world"
expected_output = "ehll owrld"
method(input_string)
# assert method(input_string) == expected_output, f"Test failed! Expected {expected_output}, but got {method(input_string)}"

# print("Test passed!")
def method(input_string):
    def sort_characters(word):
        return ''.join(sorted(word))
    
    def sort_order_string(input_string):
        words = input_string.split()
        sorted_words = [sort_characters(word) for word in words]
        return ' '.join(sorted_words)
    
    return sort_order_string(input_string)

# Test case
test_case = "Hello World"
output = method(test_case)
print(output)  # Expected output: "derWllOh"
def method():
    input_string = "test"
    
    def get_prefixes(s):
        return [s[:i] for i in range(1, len(s) + 1)]
    
    output = get_prefixes(input_string)
    return output

# # Test case
# def test_method():
#     assert method() == ['t', 'te', 'tes', 'test']
#     print("Test passed!")

# # Run the test
# test_method()

# Run the method
method()
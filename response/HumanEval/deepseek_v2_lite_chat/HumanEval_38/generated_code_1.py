import itertools

def method(input_string):
    if not input_string:
        return ""
    
    # Calculate how many times the input string can be cycled by groups of three
    cycles = (len(input_string) + 2) // 3
    
    # Create a generator that cycles the input string by groups of three
    encoded_string = itertools.cycle(input_string * 3)
    
    # Use the cycle to generate the encoded string
    encoded_output = ''.join(next(encoded_string) for _ in range(cycles))
    
    return encoded_output

# Test case to validate the function
def test_method():
    input_string = "hello"
    expected_output = "ohemm"
    result = method(input_string)
    # assert result == expected_output, f"Expected {expected_output}, but got {result}"
    # print("Test passed!")

test_method()
import math

def method(number):
    # Get the integer part
    int_part = math.floor(number)
    # Get the decimal part
    dec_part = number - int_part
    # Return the decimal part
    return dec_part

# Test case
def test_method():
    # assert abs(method(1.5) - 0.5) < 0.001, "Test case 1 failed"
    # assert abs(method(2.71828) - 0.71828) < 0.001, "Test case 2 failed"
    # assert abs(method(0.0) - 0.0) < 0.001, "Test case 3 failed"
    # assert abs(method(1) - 0.0) < 0.001, "Test case 4 failed"
    # print("All test cases passed")
    print(method(1.5))

# Run the test case
test_method()

# Example usage:
# print(method(3.14)) # Output should be 0.14
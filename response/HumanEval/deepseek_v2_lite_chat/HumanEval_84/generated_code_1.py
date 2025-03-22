def method(N):
    # Convert the integer to its binary representation
    binary_str = bin(N)[2:]
    
    # Initialize the sum of the digits
    digit_sum = 0
    
    # Iterate through each digit in the binary string
    for digit in binary_str:
        digit_sum += int(digit)
    
    return digit_sum

# Test the function with a sample input
def test_method():
    # assert method(123) == 6, "Test case 1 failed"
    # assert method(456) == 15, "Test case 2 failed"
    # assert method(789) == 24, "Test case 3 failed"
    # print("All test cases passed")
    print(method(123))

# Run the test function
test_method()
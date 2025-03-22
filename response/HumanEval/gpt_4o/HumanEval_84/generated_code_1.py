def method(N):
    # Sum the digits of N
    digit_sum = sum(int(digit) for digit in str(N))
    
    # Convert the sum to binary and remove the '0b' prefix
    output = bin(digit_sum)[2:]
    
    return output

# Test case
def test_method():
    # Test with a specific input
    N = 123  # The sum of digits is 1 + 2 + 3 = 6, and binary of 6 is '110'
    expected_output = '110'
    assert method(N) == expected_output, f"Test failed: expected {expected_output}, got {method(N)}"
    print("Test passed!")

# Run the test case
test_method()
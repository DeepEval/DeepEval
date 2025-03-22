def method(a, b):
    # Extract unit digits
    unit_digit_a = abs(a) % 10
    unit_digit_b = abs(b) % 10
    
    # Compute the product of the unit digits
    output = unit_digit_a * unit_digit_b
    
    return output

# Test case to validate the function
def test_method():
    # Test case 1: Positive numbers
    assert method(23, 45) == 15, "Test case 1 failed"
    
    # Test case 2: Negative and positive number
    assert method(-13, 29) == 27, "Test case 2 failed"
    
    # Test case 3: Both numbers negative
    assert method(-18, -74) == 32, "Test case 3 failed"
    
    # Test case 4: One number is zero
    assert method(0, 9) == 0, "Test case 4 failed"
    
    # Test case 5: Both numbers are zero
    assert method(0, 0) == 0, "Test case 5 failed"
    
    print("All test cases passed!")

# Run test case
test_method()
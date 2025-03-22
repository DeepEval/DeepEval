def method(a, b):
    # Simple function to add two numbers
    output = a + b
    return output

# Test cases to validate the method function
def test_method():
    # Test case 1: Positive numbers
    assert method(2, 3) == 5, "Test case 1 failed"
    
    # Test case 2: Negative numbers
    assert method(-1, -1) == -2, "Test case 2 failed"
    
    # Test case 3: Positive and Negative number
    assert method(5, -3) == 2, "Test case 3 failed"
    
    # Test case 4: Zero
    assert method(0, 0) == 0, "Test case 4 failed"
    
    # Test case 5: Large numbers
    assert method(1000000, 5000000) == 6000000, "Test case 5 failed"
    
    # Test case 6: Float numbers
    assert method(2.5, 3.5) == 6.0, "Test case 6 failed"
    
    print("All test cases passed!")

# Run the test cases
test_method()
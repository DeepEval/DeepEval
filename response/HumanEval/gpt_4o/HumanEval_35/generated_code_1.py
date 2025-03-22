def method(lst):
    # Check if the list is empty and handle it accordingly
    if not lst:
        return None  # or raise an exception based on your requirements
    
    # Use the built-in max function to find the maximum element
    output = max(lst)
    
    return output

# Test case to validate the method
def test_method():
    # Test with a normal list
    assert method([1, 2, 3, 4, 5]) == 5, "Test Case 1 Failed"
    
    # Test with a list with negative numbers
    assert method([-1, -2, -3, -4, -5]) == -1, "Test Case 2 Failed"
    
    # Test with a list with mixed positive and negative numbers
    assert method([-10, 5, 3, -2, 0]) == 5, "Test Case 3 Failed"
    
    # Test with a list with a single element
    assert method([10]) == 10, "Test Case 4 Failed"
    
    # Test with a list with repeating maximum elements
    assert method([3, 3, 3, 3]) == 3, "Test Case 5 Failed"
    
    # Test with an empty list
    assert method([]) == None, "Test Case 6 Failed"
    
    print("All test cases passed.")

# Run the test
test_method()
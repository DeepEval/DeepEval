import numpy as np

def method(numbers):
    # Calculate the mean of the numbers
    mean_value = np.mean(numbers)
    
    # Calculate the Mean Absolute Deviation
    mad = np.mean(np.abs(numbers - mean_value))
    
    return mad

# Test case for validation
def test_method():
    numbers = [1, 2, 3, 4, 5]
    expected_output = 1.2
    output = method(numbers)
    assert abs(output - expected_output) < 1e-9, f"Test failed: Expected {expected_output}, got {output}"
    print("Test passed!")

# Run the test case
test_method()
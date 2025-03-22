import numpy as np

def method(numbers):
    # Calculate mean of the input numbers
    mean = np.mean(numbers)
    
    # Calculate the absolute differences from the mean
    absolute_diffs = np.abs(numbers - mean)
    
    # Calculate Mean Absolute Deviation
    mad = np.mean(absolute_diffs)
    
    return mad

# Test case for validation
if __name__ == "__main__":
    test_numbers = [1, 2, 3, 4, 5]
    output = method(np.array(test_numbers))
    print("Mean Absolute Deviation:", output)
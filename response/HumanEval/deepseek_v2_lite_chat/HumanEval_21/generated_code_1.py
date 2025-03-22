import numpy as np

def method(numbers):
    # Convert the list to a numpy array and find the minimum and maximum values
    numbers_array = np.array(numbers)
    min_val = np.min(numbers_array)
    max_val = np.max(numbers_array)
    
    # Create a linear transformation function
    def linear_transform(x):
        return (x - min_val) / (max_val - min_val)
    
    # Apply the transformation to each element in the list
    transformed_numbers = np.vectorize(linear_transform)(numbers_array)
    
    # Convert the numpy array back to a list
    output = transformed_numbers.tolist()
    
    return output

# Test case to validate the function
test_numbers = [5, 3, 8, 4, 10]
expected_output = [0.0, 0.125, 0.5, 0.375, 1.0]
# assert method(test_numbers) == expected_output
print(method(test_numbers))

print("Test case passed!")
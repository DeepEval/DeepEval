import numpy as np

def method():
    # Create two tensors (numpy arrays) for demonstration
    tensor_a = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_b = np.array([[7, 8, 9], [10, 11, 12]])

    # Performing operations on the tensors
    addition = tensor_a + tensor_b            # Element-wise addition
    subtraction = tensor_a - tensor_b         # Element-wise subtraction
    multiplication = tensor_a * tensor_b      # Element-wise multiplication
    division = tensor_a / tensor_b            # Element-wise division
    reshaped_tensor = tensor_a.reshape(3, 2)  # Reshaping tensor_a to (3, 2)

    # Combine results into a dictionary for output
    output = {
        'addition': addition,
        'subtraction': subtraction,
        'multiplication': multiplication,
        'division': division,
        'reshaped_tensor': reshaped_tensor
    }

    return output

# Call the method for validation
if __name__ == "__main__":
    result = method()
    print(result)
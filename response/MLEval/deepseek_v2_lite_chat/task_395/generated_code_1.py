import numpy as np


def method():
    # Define the tensors
    tensor1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    tensor2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # Element-wise multiplication
    result_mul = np.multiply(tensor1, tensor2)
    
    # Element-wise addition
    result_add = np.add(tensor1, tensor2)
    
    # Matrix multiplication
    result_matmul = np.matmul(tensor1, tensor2.T)
    
    # Concatenate the results into a single tensor
    final_output = np.concatenate((result_mul, result_add, result_matmul), axis=1)

    return final_output

# Call the function for validation
output = method()
print(output)
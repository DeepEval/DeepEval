import numpy as np
import torch

def pyramid_combination(values, weight_floor, weight_ceil):
    n = len(weight_floor)
    n_values = 2 ** n
    
    if len(values)!= n_values:
        raise ValueError("Length of values does not match 2^n")
    
    # Check if the shapes of values, weight_floor, and weight_ceil are compatible
    for i in range(n_values):
        value = values[i]
        weight_floor_i = weight_floor[i % len(weight_floor)]
        weight_ceil_i = weight_ceil[i % len(weight_ceil)]
        
        if value.shape!= weight_floor_i.shape or value.shape!= weight_ceil_i.shape:
            raise ValueError("Shape mismatch between values and weights")
    
    # Perform linear interpolation
    output = []
    for i in range(n_values):
        weight_floor_i = weight_floor[i % len(weight_floor)]
        weight_ceil_i = weight_ceil[i % len(weight_ceil)]
        value = values[i]
        
        interpolated_value = weight_floor_i * value + weight_ceil_i * value
        
        output.append(interpolated_value)
    
    return torch.stack(output)

if __name__ == "__main__":
    # Create sample inputs
    values = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]
    weight_floor = [torch.tensor([[0.5, 0.5], [0.5, 0.5]])]
    weight_ceil = [torch.tensor([[0.5, 0.5], [0.5, 0.5]])]
    
    # Call the function and print the result
    result = pyramid_combination(values, weight_floor, weight_ceil)
    print(result)
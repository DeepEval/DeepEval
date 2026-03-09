import torch

def pyramid_combination(values, weight_floor, weight_ceil):
    n = len(weight_floor)
    if len(values) != 2**n:
        raise ValueError("Length of values must be 2^n where n is the length of weight_floor and weight_ceil")
    
    for wf, wc in zip(weight_floor, weight_ceil):
        if wf.shape != wc.shape:
            raise ValueError("Shapes of weight_floor and weight_ceil must match")

    result = torch.zeros_like(values[0])
    
    for i in range(2**n):
        weight_product = torch.ones_like(values[0])
        for j in range(n):
            if (i >> j) & 1:
                weight_product *= weight_ceil[j]
            else:
                weight_product *= weight_floor[j]
        result += values[i] * weight_product

    return result

if __name__ == "__main__":
    # Define a 2D hypercube (4 corners for 2D)
    values = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
        torch.tensor([[13.0, 14.0], [15.0, 16.0]])
    ]
    
    # Define weight_floor and weight_ceil for each dimension
    weight_floor = [
        torch.tensor([[0.2, 0.2], [0.2, 0.2]]),
        torch.tensor([[0.3, 0.3], [0.3, 0.3]])
    ]
    
    weight_ceil = [
        torch.tensor([[0.8, 0.8], [0.8, 0.8]]),
        torch.tensor([[0.7, 0.7], [0.7, 0.7]])
    ]
    
    # Call the function and print the result
    result = pyramid_combination(values, weight_floor, weight_ceil)
    print("Interpolated result:\n", result)
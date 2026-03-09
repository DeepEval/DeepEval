import torch

def pyramid_combination(values, weight_floor, weight_ceil):
    n = len(weight_floor)
    if len(values) != 2 ** n:
        raise ValueError("Length of values must be 2^n")
    if len(weight_floor) != len(weight_ceil):
        raise ValueError("Lengths of weight_floor and weight_ceil must be the same")
    if len(weight_floor) != n:
        raise ValueError("Length of weight_floor must be equal to n")
    
    # Check that the shapes of the inputs are compatible
    for value, weight_floor, weight_ceil in zip(values, weight_floor, weight_ceil):
        if value.shape[-1] != 2 ** n:
            raise ValueError("Shape of value must match 2^n")
    
    # Perform linear interpolation
    output = []
    for i in range(len(values)):
        output.append(weight_floor[i] * values[i] + weight_ceil[i] * values[i])
    return torch.stack(output, dim=0)

if __name__ == "__main__":
    # Sample input values
    values = [torch.rand(10, 2**5) for _ in range(4)]
    
    # Weight floors and ceilings
    weight_floor = [torch.rand(10, 2**i) for i in range(5)]
    weight_ceil = [torch.rand(10, 2**i) for i in range(5)]
    
    # Call the function and print results
    result = pyramid_combination(values, weight_floor, weight_ceil)
    print("Result:", result)
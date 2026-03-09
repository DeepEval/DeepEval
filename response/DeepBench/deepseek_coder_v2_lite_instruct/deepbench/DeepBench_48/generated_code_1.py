import torch

def pyramid_combination(values, weight_floor, weight_ceil):
    # Check if the length of values matches the expected 2^n based on the dimension of the weights
    n = len(weight_floor)
    if len(values) != 2**n:
        raise ValueError("The number of values must be 2^n where n is the dimension of the space.")
    
    # Check if the shapes of the inputs are compatible
    for value, weight_floor_elem, weight_ceil_elem in zip(values, weight_floor, weight_ceil):
        if value.shape != weight_floor_elem.shape or value.shape != weight_ceil_elem.shape:
            raise ValueError("The shapes of values, weight_floor, and weight_ceil must be compatible.")
    
    # Perform linear interpolation
    interpolation = torch.zeros_like(values[0])
    for i in range(2**n):
        weight = 1.0
        index = i
        for j in range(n):
            if index % 2 == 1:
                weight *= weight_floor[j]
            else:
                weight *= weight_ceil[j]
            index //= 2
        interpolation += values[i] * weight
    
    return interpolation

if __name__ == "__main__":
    # Example usage
    values = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])]
    weight_floor = [torch.tensor(0.5), torch.tensor(0.5)]
    weight_ceil = [torch.tensor(0.5), torch.tensor(0.5)]
    
    result = pyramid_combination(values, weight_floor, weight_ceil)
    print(result)  # Expected output: tensor([0.5000, 0.5000])
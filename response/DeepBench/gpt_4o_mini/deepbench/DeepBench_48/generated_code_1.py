import torch

def pyramid_combination(values, weight_floor, weight_ceil):
    n = len(weight_floor)
    expected_length = 2 ** n

    if len(values) != expected_length:
        raise ValueError(f"Expected {expected_length} values, got {len(values)}")
    
    for v in values:
        if v.shape != weight_floor[0].shape or v.shape != weight_ceil[0].shape:
            raise ValueError("Shapes of values and weights must be compatible")
    
    weights = [torch.unsqueeze(wf, 0) * torch.unsqueeze(wc, 0) for wf, wc in zip(weight_floor, weight_ceil)]
    
    result = sum(v * w for v, w in zip(values, weights))
    return result

if __name__ == "__main__":
    values = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0])]
    weight_floor = [torch.tensor([0.25]), torch.tensor([0.25]), torch.tensor([0.25]), torch.tensor([0.25])]
    weight_ceil = [torch.tensor([0.75]), torch.tensor([0.75]), torch.tensor([0.75]), torch.tensor([0.75])]

    result = pyramid_combination(values, weight_floor, weight_ceil)
    print(result)
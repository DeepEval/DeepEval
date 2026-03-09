import torch

def silu(input_tensor, inplace=False):
    if inplace:
        return input_tensor.mul_(input_tensor.sigmoid())
    else:
        return input_tensor * input_tensor.sigmoid()

if __name__ == "__main__":
    # Create sample input values
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    
    # Call the function and print the results
    output_tensor = silu(input_tensor)
    print("Output tensor (non-inplace):", output_tensor)
    
    # Call the function with inplace=True and print the results
    output_tensor_inplace = silu(input_tensor, inplace=True)
    print("Output tensor (inplace):", output_tensor_inplace)
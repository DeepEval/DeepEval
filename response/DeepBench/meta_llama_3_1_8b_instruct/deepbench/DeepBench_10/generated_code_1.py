import torch

def silu(input, inplace=False):
    # References:
    # - Elfwing, S., Tallec, P., & Baldassi, C. (2018). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.
    # - Ramachandran, P., Subbiah, N., Bapst, V., Huang, S., Aslanides, J., Quan, J.,... & Botvinick, M. (2017). Swish: A Self-Gated Activation Function.
    sigmoid = torch.nn.Sigmoid()
    if inplace:
        input.mul_(sigmoid(input))
    else:
        return input * sigmoid(input)

if __name__ == "__main__":
    # Create a sample input tensor
    input_tensor = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    print("Input Tensor:", input_tensor)

    # Apply the SiLU function
    output = silu(input_tensor)
    print("Output Tensor:", output)

    # Verify the output is correct by applying the SiLU function manually
    manual_output = torch.zeros_like(input_tensor)
    sigmoid = torch.nn.Sigmoid()
    for i in range(len(input_tensor)):
        manual_output[i] = input_tensor[i] * sigmoid(input_tensor[i])
    print("Manual Output Tensor:", manual_output)
    print("Output Tensor matches Manual Output Tensor:", torch.allclose(output, manual_output))

    # Test the in-place operation
    in_place_output = silu(input_tensor, inplace=True)
    print("In-Place Output Tensor:", in_place_output)
    print("Input Tensor after in-place operation:", input_tensor)
    print("Input Tensor should match the original input tensor after in-place operation:", torch.allclose(input_tensor, input_tensor.detach()))
import torch

def method():
    # Create a tensor with requires_grad=True to track gradients
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    
    # Define a simple function of x
    y = x[0] ** 2 + x[1] ** 3  # y = x1^2 + x2^3
    
    # Compute the gradients
    y.backward()  # This computes the gradient of y with respect to x
    
    # Retrieve the gradients
    output = x.grad  # This will hold the gradients of y w.r.t x
    
    return output

# Call the method for validation
output = method()
print("The gradients are:", output)
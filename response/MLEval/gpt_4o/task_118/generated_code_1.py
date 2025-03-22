import torch

def method():
    # Create a tensor with requires_grad=True to track computations
    x = torch.tensor([2.0], requires_grad=True)
    
    # Define a simple function using this tensor
    y = x ** 2  # Example function: y = x^2
    
    # Compute the gradient (dy/dx)
    y.backward()
    
    # Access the gradient of x
    output = x.grad
    
    return output

# Call the method for validation
gradient = method()
print("The gradient is:", gradient)
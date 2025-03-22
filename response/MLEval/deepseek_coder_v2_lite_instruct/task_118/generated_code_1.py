import torch

def method():
    # Define the input tensor
    x = torch.tensor(3.0, requires_grad=True)
    
    # Define the function y = x^2
    y = x**2
    
    # Compute the gradient of y with respect to x
    y.backward()
    
    # Retrieve the gradient
    gradient = x.grad
    
    # Return the gradient
    return gradient

# Call the method and print the result
output = method()
print(output)  # Output should be 6.0, which is the gradient of y = x^2 at x = 3.0
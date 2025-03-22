import torch

def method():
  # Example usage:
  x = torch.randn(2, 2, requires_grad=True)
  y = torch.ones(2, 2)

  # Calculate the gradient
  z = torch.matmul(x, y)
  z.backward()

  # Print the gradient
  print(x.grad)

  # Return the gradient if needed
  return x.grad

# Call the method for validation
output = method()
print(output)
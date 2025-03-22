import torch
import torch.nn as nn

def method():
    # Define a simple linear model
    model = nn.Linear(10, 5)

    # Create some sample input data
    input_data = torch.randn(1, 10)

    # Calculate the output
    output = model(input_data)

    # Calculate the gradient of the output with respect to the input
    gradient = torch.autograd.grad(outputs=output, inputs=input_data, create_graph=True)[0]

    return gradient

# Call the method and print the output
gradient = method()
print(gradient)
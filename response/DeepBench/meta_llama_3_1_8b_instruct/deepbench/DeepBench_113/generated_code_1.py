import torch
import torch.nn as nn

class Kron:
    def __init__(self, kronecker_factors):
        self.kronecker_factors = kronecker_factors

def init_from_model(model, device):
    if isinstance(model, torch.nn.Module):
        parameters = model.parameters()
    elif isinstance(model, (tuple, list)):
        parameters = model
    else:
        raise ValueError("Invalid model type")

    kronecker_factors = []
    for param in parameters:
        shape = param.shape
        if len(shape) == 1:  # bias
            kronecker_factors.append(torch.zeros((shape[0], shape[0])))
        elif len(shape) == 2:  # fully connected or convolutional layers
            kronecker_factors.append(torch.zeros((shape[0], shape[0])))
            kronecker_factors.append(torch.zeros((shape[1], shape[1])))
        else:
            raise ValueError("Invalid parameter shape")

    return Kron(kronecker_factors)

if __name__ == "__main__":
    # create a sample model
    model = nn.Linear(5, 10)
    parameters = model.parameters()

    # call the function
    kron = init_from_model(model, torch.device("cpu"))

    # print the results
    for i, kronecker_factor in enumerate(kron.kronecker_factors):
        print(f"Kronecker factor {i+1}: {kronecker_factor.shape}")
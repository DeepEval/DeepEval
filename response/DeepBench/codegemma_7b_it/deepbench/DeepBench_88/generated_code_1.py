import torch
from torch.nn import Module
from functorch import vmap
import functorch.optim as optim
import math
from typing import Callable, Iterable
import copy
from functools import partial
from torch.autograd import Function
from torch.linalg import LinAlgError

from functools import partial
from torch.autograd import Function
from torch.linalg import LinAlgError

class Laplace:
    def __init__(self, model: Module, likelihood: str, subset_of_weights: str = SubsetOfWeights.LAST_LAYER, hessian_structure: str = HessianStructure.KRON):
        if subset_of_weights == "subnetwork" and hessian_structure not in ["full", "diag"]:
            raise ValueError( "Subnetwork Laplace requires a full or diagonal Hessian approximation!" )
        self.model = model
        self.likelihood = likelihood
        self.subset_of_weights = subset_of_weights
        self.hessian_structure = hessian_structure

    def __call__(self, model_params):
        # Your code here

        return output

if __name__ == "__main__":
    # Sample input values
    model = torch.nn.Linear(10, 5)
    likelihood = "classification"

    # Call the function
    laplace = Laplace(model, likelihood)

    # Print the results
    print(laplace)
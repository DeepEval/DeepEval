import torch
from torch.nn import Module
from torch.distributions import Likelihood
from torch.distributions.laplace import ParametricLaplace
from torch.distributions.laplace import HessianStructure

def Laplace(model, likelihood, subset_of_weights=SubsetOfWeights.LAST_LAYER, hessian_structure=HessianStructure.KRON):
    # Check if the input is a string
    if isinstance(subset_of_weights, str):
        # Convert the string to a SubsetOfWeights object
        subset_of_weights = SubsetOfWeights[subset_of_weights]
    # Check if the input is a string
    if isinstance(hessian_structure, str):
        # Convert the string to a HessianStructure object
        hessian_structure = HessianStructure[hessian_structure]

    # Create a dictionary mapping subclass keys to subclasses of ParametricLaplace
    subclass_map = {
        'classification': ClassificationLaplace,
        'regression': RegressionLaplace
    }

    # Instantiate the chosen subclass with additional arguments
    subclass_instance = subclass_map[likelihood](model, subset_of_weights, hessian_structure)

    return subclass_instance

# Runnable example
if __name__ == "__main__":
    # Create a sample model
    model = torch.nn.Linear(5, 3)

    # Create a sample likelihood
    likelihood = Likelihood('classification')

    # Create a sample subset of weights
    subset_of_weights = SubsetOfWeights('last_layer')

    # Create a sample hessian structure
    hessian_structure = HessianStructure('kron')

    # Call the function
    laplace = Laplace(model, likelihood, subset_of_weights, hessian_structure)

    # Print the results
    print(laplace)
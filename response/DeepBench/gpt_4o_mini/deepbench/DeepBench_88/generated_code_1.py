import torch
import torch.nn as nn

class ParametricLaplace:
    def __init__(self, model, likelihood, subset_of_weights, hessian_structure):
        self.model = model
        self.likelihood = likelihood
        self.subset_of_weights = subset_of_weights
        self.hessian_structure = hessian_structure

def Laplace(model, likelihood, subset_of_weights='last_layer', hessian_structure='kron'):
    if subset_of_weights == "subnetwork" and hessian_structure not in ["full", "diag"]:
        raise ValueError("Subnetwork Laplace requires a full or diagonal Hessian approximation!")

    subclasses = {
        'classification': ParametricLaplace,
        'regression': ParametricLaplace,
        'last_layer': ParametricLaplace,
        'subnetwork': ParametricLaplace,
        'all': ParametricLaplace
    }

    subclass_key = likelihood if isinstance(likelihood, str) else 'classification'
    subclass_key = subset_of_weights if isinstance(subset_of_weights, str) else 'last_layer'

    laplace_instance = subclasses[subclass_key](model, likelihood, subset_of_weights, hessian_structure)
    return laplace_instance

if __name__ == "__main__":
    model = nn.Linear(10, 2)
    likelihood = 'classification'
    subset_of_weights = 'last_layer'
    hessian_structure = 'kron'

    laplace_instance = Laplace(model, likelihood, subset_of_weights, hessian_structure)
    print(f"Created ParametricLaplace instance: {laplace_instance}")
    print(f"Model: {laplace_instance.model}, Likelihood: {laplace_instance.likelihood}, "
          f"Subset of Weights: {laplace_instance.subset_of_weights}, "
          f"Hessian Structure: {laplace_instance.hessian_structure}")
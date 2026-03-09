import torch
from torch.nn import Module, Linear, Conv2d
from typing import Union, Dict, Any

class SubsetOfWeights:
    LAST_LAYER = 'last_layer'
    SUBNETWORK = 'subnetwork'
    ALL = 'all'

class HessianStructure:
    KRON = 'kron'
    DIAG = 'diag'
    FULL = 'full'
    LOWRANK = 'lowrank'

class ParametricLaplace:
    def __init__(self, model: Module, likelihood: str, subset_of_weights: str, hessian_structure: str):
        self.model = model
        self.likelihood = likelihood
        self.subset_of_weights = subset_of_weights
        self.hessian_structure = hessian_structure

class ClassificationLaplace(ParametricLaplace):
    def __init__(self, model: Module, likelihood: str, subset_of_weights: str, hessian_structure: str):
        super().__init__(model, likelihood, subset_of_weights, hessian_structure)

class RegressionLaplace(ParametricLaplace):
    def __init__(self, model: Module, likelihood: str, subset_of_weights: str, hessian_structure: str):
        super().__init__(model, likelihood, subset_of_weights, hessian_structure)

def Laplace(model: Module, likelihood: str, subset_of_weights: SubsetOfWeights = SubsetOfWeights.LAST_LAYER, hessian_structure: HessianStructure = HessianStructure.KRON) -> ParametricLaplace:
    if subset_of_weights == SubsetOfWeights.SUBNETWORK and hessian_structure not in [HessianStructure.FULL, HessianStructure.DIAG]:
        raise ValueError("Subnetwork Laplace requires a full or diagonal Hessian approximation!")

    subclasses_mapping: Dict[str, Any] = {
        'classification': ClassificationLaplace,
        'regression': RegressionLaplace
    }

    if subset_of_weights == SubsetOfWeights.SUBNETWORK:
        return ClassificationLaplace(model, likelihood, subset_of_weights, hessian_structure)
    else:
        subclass = subclasses_mapping.get(likelihood, ParametricLaplace)
        return subclass(model, likelihood, subset_of_weights, hessian_structure)

if __name__ == "__main__":
    # Example usage
    model = Linear(10, 1)
    laplace_obj = Laplace(model, 'classification')
    print(laplace_obj.likelihood)  # Output: classification
    print(laplace_obj.subset_of_weights)  # Output: last_layer
    print(laplace_obj.hessian_structure)  # Output: kron
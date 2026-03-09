import torch
from torch import nn
from torch.distributions import Laplace
from enum import Enum

class SubsetOfWeights(Enum):
    LAST_LAYER = 1
    SUBNETWORK = 2
    ALL = 3

class HessianStructure(Enum):
    DIAG = 1
    KRONECKER = 2
    FULL = 3
    LOWRANK = 4

class ParametricLaplace:
    pass

class ClassificationLaplace(ParametricLaplace):
    pass

class RegressionLaplace(ParametricLaplace):
    pass

class SubnetworkLaplace(ParametricLaplace):
    pass

def Laplace(model, likelihood, subset_of_weights=SubsetOfWeights.LAST_LAYER, hessian_structure=HessianStructure.KRON):
    if subset_of_weights == SubsetOfWeights.SUBNETWORK and hessian_structure not
  in [HessianStructure.FULL, HessianStructure.DIAG]:
        raise ValueError("Subnetwork Laplace requires a full or diagonal Hessian
  approximation!")

    subclasses = {
        'classification': ClassificationLaplace,
       'regression': RegressionLaplace,
        'last_layer': ClassificationLaplace,
       'subnetwork': SubnetworkLaplace,
        'all': ClassificationLaplace
    }

    subclass = subclasses.get(str(likelihood), ClassificationLaplace)

    return subclass(model, subset_of_weights, hessian_structure)

if __name__ == "__main__":
    model = nn.Linear(5, 5)
    likelihood = 'classification'
    subset_of_weights = SubsetOfWeights.LAST_LAYER
    hessian_structure = HessianStructure.KRON

    laplace = Laplace(model, likelihood, subset_of_weights, hessian_structure)
    print(laplace)

    laplace = Laplace(model, likelihood, SubsetOfWeights.SUBNETWORK, HessianStructure.DIAG)
    print(laplace)


    laplace = Laplace(model, likelihood, SubsetOfWeights.SUBNETWORK, HessianStructure.KRON)
    print(laplace)
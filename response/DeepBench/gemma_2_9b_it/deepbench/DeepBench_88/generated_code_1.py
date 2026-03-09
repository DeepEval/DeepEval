from typing import Union

from torch import nn
from parametric_laplace import (
    ParametricLaplace,
    SubsetOfWeights,
    HessianStructure,
    Likelihood,
)

def Laplace(
    model: nn.Module,
    likelihood: Union[Likelihood, str],
    subset_of_weights: Union[SubsetOfWeights, str] = SubsetOfWeights.LAST_LAYER,
    hessian_structure: Union[HessianStructure, str] = HessianStructure.KRON,
) -> ParametricLaplace:
    if subset_of_weights == "subnetwork" and hessian_structure not in ["full", "diag"]:
        raise ValueError(
            "Subnetwork Laplace requires a full or diagonal Hessian approximation!"
        )
    subclass_dict = {
        SubsetOfWeights.LAST_LAYER: {
            HessianStructure.DIAG: ParametricLaplace.DiagonalLastLayer,
            HessianStructure.KRON: ParametricLaplace.KronLastLayer,
            HessianStructure.FULL: ParametricLaplace.FullLastLayer,
            HessianStructure.LOWRANK: ParametricLaplace.LowRankLastLayer,
        },
        SubsetOfWeights.SUBNETWORK: {
            HessianStructure.DIAG: ParametricLaplace.DiagonalSubnetwork,
            HessianStructure.KRON: ParametricLaplace.KronSubnetwork,
            HessianStructure.FULL: ParametricLaplace.FullSubnetwork,
            HessianStructure.LOWRANK: ParametricLaplace.LowRankSubnetwork,
        },
        SubsetOfWeights.ALL: {
            HessianStructure.DIAG: ParametricLaplace.DiagonalAll,
            HessianStructure.KRON: ParametricLaplace.KronAll,
            HessianStructure.FULL: ParametricLaplace.FullAll,
            HessianStructure.LOWRANK: ParametricLaplace.LowRankAll,
        },
    }
    return subclass_dict[subset_of_weights][hessian_structure](model, likelihood)

if __name__ == "__main__":
    model = nn.Linear(10, 5)
    likelihood = "classification" 
    subset_of_weights = SubsetOfWeights.LAST_LAYER
    hessian_structure = HessianStructure.KRON
    laplace = Laplace(model, likelihood, subset_of_weights, hessian_structure)
    print(laplace)
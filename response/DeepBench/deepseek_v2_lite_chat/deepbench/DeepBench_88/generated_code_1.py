import torch
from torch.optim.optimizer import Optimizer

class DiagLaplace(Optimizer):
    def __init__(self, params, hessian):
        defaults = dict(hessian=hessian)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                hessian = group['hessian']
                grad = p.grad.data
                state = self.state[p]

                # Lazy initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['hessian'] = group['hessian']

                # Calculate parameters
                if len(hessian) == 1:
                    diag = grad.view(-1).diag()
                    dp = grad.view(1, -1) * grad.view(-1, 1)
                    dp[diag > 0] = grad[diag > 0] / diag[diag > 0].view(-1, 1)
                    dp[diag <= 0] = 0
                else:
                    dp = hessian[0]

                # Decay the inverse Hessian approximation
                decay_rate = 1.0 - 1.0 / (2.0 * state['step'] + 1)
                dp = decay_rate * dp

                p.add_(dp)

        return loss

class KronLaplace(Optimizer):
    def __init__(self, params, hessian):
        defaults = dict(hessian=hessian)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                hessian = group['hessian']
                grad = p.grad.data
                state = self.state[p]

                # Lazy initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['hessian'] = group['hessian']

                # Calculate parameters
                dp = hessian @ grad

                # Decay the inverse Hessian approximation
                decay_rate = 1.0 - 1.0 / (2.0 * state['step'] + 1)
                dp = decay_rate * dp

                p.add_(dp)

        return loss

def Laplace(model, likelihood, subset_of_weights=SubsetOfWeights.LAST_LAYER, hessian_structure=HessianStructure.KRON):
    if subset_of_weights == SubsetOfWeights.SUBNETWORK and hessian_structure not in ["full", "diag"]:
        raise ValueError("Subnetwork Laplace requires a full or diagonal Hessian approximation!")

    if subset_of_weights == SubsetOfWeights.LAST_LAYER:
        return DiagLaplace(model.parameters(), getattr(hessian_structure, 'value', None))
    elif subset_of_weights == SubsetOfWeights.SUBNETWORK:
        return KronLaplace(model.parameters(), getattr(hessian_structure, 'value', None))
    else:
        raise ValueError("Unsupported subset of weights")

if __name__ == "__main__":
    import torch.optim as optim
    import numpy as np

    # Sample input values
    model = torch.nn.Sequential(torch.nn.Linear(10, 50), torch.nn.ReLU(), torch.nn.Linear(50, 2))
    likelihood = 'classification'
    subset_of_weights = SubsetOfWeights.LAST_LAYER
    hessian_structure = HessianStructure.KRON

    # Call the function and print the results
    laplace = Laplace(model, likelihood, subset_of_weights, hessian_structure)
    print(laplace.param_groups)
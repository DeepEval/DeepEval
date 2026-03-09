import numpy as np
import jax.numpy as jnp
from torch import Tensor
from typing import Optional
from scipy.linalg import norm as scipy_norm


def _flat_autograd_norm(x: Tensor, p: float = 2, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    return jnp.where(x.is_array, jnp.linalg.norm(x, ord=p, axis=dim, keepdims=keepdim), jnp.linalg.norm(x.item(), ord=p))


def norm(x: Tensor, interface: str = "default", p: Optional[float] = None, axis: Optional[int] = None, **kwargs) -> Tensor:
    if interface == "torch":
        axis = axis if axis is not None else list(range(x.ndim))
        axis = tuple(axis)
        return scipy_norm(x.flatten(), ord=p) if p is None else scipy_norm(x.flatten(), ord=p, axis=axis)
    elif interface == "jax":
        return jnp.linalg.norm(x, ord=p)
    elif interface == "autograd":
        return _flat_autograd_norm(x, p=p, axis=axis)
    else:
        return scipy_norm(x, ord=p)


if __name__ == "__main__":
    import torch
    x = torch.tensor([[1, 2], [3, 4]])
    p = 2
    print(norm(x, "torch", p=p))  # Should print out 5.830951894845301
    print(norm(x, "torch", p=None))  # Should print out 7.483314773547882
    print(norm(x, "jax", p=2))  # Should print out 5.830951894845301
    print(norm(x, "autograd", p=2))  # Should print out 5.830951894845301
import scipy as sp
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch
from scipy import linalg
from jax import vmap
from autograd import numpy as npa

def norm(tensor, which="scipy", axis=None, **kwargs):
    if which == "scipy":
        return linalg.norm(tensor, **kwargs)
    elif which == "jax":
        return jnp.linalg.norm(tensor, **kwargs)
    elif which == "tensorflow":
        return tf.norm(tensor, **kwargs)
    elif which == "torch":
        if axis is not None:
            axis = axis if isinstance(axis, list) else [axis]
            tensor = tensor.permute(axis + [i for i in range(tensor.ndim) if i not in axis])
        return torch.norm(tensor, **kwargs)
    elif which == "autograd":
        if axis is None:
            return npa.linalg.norm(tensor, **kwargs)
        elif isinstance(tensor, npa.ndarray):
            return _flat_autograd_norm(tensor, axis, **kwargs)
        else:
            raise ValueError("axis must be None for non-numpy arrays")
    else:
        raise ValueError("Invalid interface")

def _flat_autograd_norm(tensor, axis, ord=None, keepdims=False):
    if ord is None:
        ord = 2
    if axis is None:
        axis = tuple(range(tensor.ndim))
    shape = list(tensor.shape)
    flat_shape = tuple(np.prod(shape[i] if i not in axis else 1) for i in range(tensor.ndim))
    flat_tensor = tensor.reshape(flat_shape)
    return npa.linalg.norm(flat_tensor, ord, keepdims)

if __name__ == "__main__":
    import jax.numpy as jnp
    import torch
    import tensorflow as tf
    import numpy as np

    # Create sample input values
    tensor_jax = jnp.array([1, 2, 3])
    tensor_torch = torch.tensor([1, 2, 3])
    tensor_tf = tf.constant([1, 2, 3])
    tensor_np = np.array([1, 2, 3])
    tensor_scipy = np.array([1, 2, 3])

    # Call the function and print the results
    print(norm(tensor_jax, "jax"))
    print(norm(tensor_torch, "torch"))
    print(norm(tensor_tf, "tensorflow"))
    print(norm(tensor_np, "scipy"))
    print(norm(tensor_scipy, "scipy"))
    print(norm(tensor_np, "autograd"))
    print(norm(tensor_np, "autograd", axis=0))
    print(norm(tensor_np, "autograd", axis=0, ord=1))
import numpy as np
from scipy.linalg import norm

def norm(tensor, interface="scipy", axis=None, **kwargs):
    if interface == "jax":
        import jax.numpy as jnp
        return jnp.linalg.norm(tensor, axis=axis, **kwargs)
    elif interface == "tensorflow":
        import tensorflow as tf
        return tf.linalg.norm(tensor, axis=axis, **kwargs)
    elif interface == "torch":
        import torch
        if axis is not None:
            axis = tuple(axis)
        return torch.norm(tensor, dim=axis, **kwargs)
    elif interface == "autograd":
        import autograd.numpy as anp
        return anp.linalg.norm(tensor, axis=axis, **kwargs)
    else:
        return norm(tensor, axis=axis, **kwargs)

if __name__ == "__main__":
    # Example usage
    tensor = np.array([[1, 2], [3, 4]])
    print(norm(tensor, interface="scipy"))  # Output: 5.477225575051661
    print(norm(tensor, interface="jax"))  # Output: 5.477225575051661
    print(norm(tensor, interface="tensorflow"))  # Output: 5.477225575051661
    print(norm(tensor, interface="torch"))  # Output: 5.477225575051661
    print(norm(tensor, interface="autograd"))  # Output: 5.477225575051661
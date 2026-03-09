import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import torch

def norm(tensor, interface="scipy", axis=None):
    if interface == "jax":
        if axis is not None:
            axis = tuple(axis)  # Ensure axis is a tuple
        return jnp.linalg.norm(tensor, axis=axis)
    elif interface == "tensorflow":
        return tf.linalg.norm(tensor, axis=axis)
    elif interface == "torch":
        if axis is not None:
            axis = tuple(axis)  # Ensure axis is a tuple
        if "autograd" in interface:
            return _flat_autograd_norm(tensor, axis=axis)
        return torch.linalg.norm(tensor, dim=axis)
    else:  # Default to scipy
        return np.linalg.norm(tensor, axis=axis)

def _flat_autograd_norm(tensor, axis=None):
    if axis is not None:
        tensor = tensor.flatten(axis)
    return np.linalg.norm(tensor)

if __name__ == "__main__":
    # Example usage
    tensor = np.array([[1, 2, 3], [4, 5, 6]])
    
    print("Using jax interface:", norm(tensor, interface="jax"))
    print("Using tensorflow interface:", norm(tensor, interface="tensorflow"))
    print("Using torch interface:", norm(tensor, interface="torch"))
    print("Using scipy interface:", norm(tensor, interface="scipy"))
    print("Using autograd interface:", norm(tensor, interface="autograd"))
import jax.numpy as jnp
import tensorflow as tf
import torch
import autograd.numpy as anp
import numpy as np

def norm(tensor, like="scipy", axis=None):
  if like == "jax":
    return jnp.linalg.norm(tensor, axis=axis)
  elif like == "tensorflow":
    return tf.norm(tensor, axis=axis)
  elif like == "torch":
    if axis is not None:
      axis = [a.item() for a in axis]
    return torch.norm(tensor, dim=axis)
  elif like == "autograd":
    if isinstance(tensor, anp.core.Basic).dtype.name[:3] == "int":
      return anp.linalg.norm(tensor)
    elif isinstance(tensor, anp.core.Basic).dtype.name[:3] == "float":
      if len(tensor.shape) in [0, 1]:
        return anp.linalg.norm(tensor)
      else:
        return _flat_autograd_norm(tensor)
  elif like == "scipy":
    return np.linalg.norm(tensor)

def _flat_autograd_norm(x):
  flat_x = x.reshape((np.prod(x.shape),))
  return np.linalg.norm(flat_x)

if __name__ == "__main__":
  tensor_jax = jnp.array([1, 2, 3])
  tensor_tf = tf.constant([1, 2, 3])
  tensor_torch = torch.tensor([1, 2, 3])
  tensor_autograd = anp.array([1, 2, 3])

  print("JAX norm:", norm(tensor_jax))
  print("TensorFlow norm:", norm(tensor_tf))
  print("Torch norm:", norm(tensor_torch))
  print("Autograd norm:", norm(tensor_autograd))
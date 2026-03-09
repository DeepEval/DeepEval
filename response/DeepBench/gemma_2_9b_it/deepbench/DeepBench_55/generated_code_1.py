import jax
import jax.numpy as jnp
import tensorflow as tf
import torch
import autograd
import numpy as np

def norm(tensor, which='scipy', **kwargs):
  if which == 'jax':
    return jnp.linalg.norm(tensor, **kwargs)
  elif which == 'tensorflow':
    return tf.norm(tensor, **kwargs)
  elif which == 'torch':
    if 'axis' in kwargs:
      kwargs['dim'] = kwargs.pop('axis')
    return torch.norm(tensor, **kwargs)
  elif which == 'autograd':
    if isinstance(tensor, autograd.ndarray) and len(tensor.shape) == 1:
      return _flat_autograd_norm(tensor)
    else:
      return autograd.numpy.linalg.norm(tensor, **kwargs)
  elif which == 'scipy':
    return np.linalg.norm(tensor, **kwargs)
  else:
    raise ValueError(f"Unsupported interface: {which}")

def _flat_autograd_norm(x):
  return autograd.numpy.sqrt(autograd.numpy.sum(x**2))

if __name__ == "__main__":
  tensor = jnp.array([[1, 2], [3, 4]])
  print(f'JAX norm: {norm(tensor, which="jax")}')
  print(f'TensorFlow norm: {norm(tf.convert_to_tensor(tensor), which="tensorflow")}')
  print(f'Torch norm: {norm(torch.from_numpy(tensor), which="torch")}')
  print(f'Autograd norm: {norm(autograd.numpy.array(tensor), which="autograd")}')
  print(f'SciPy norm: {norm(tensor, which="scipy")}')
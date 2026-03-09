def norm(tensor, interface='scipy', axis=None, **kwargs):
    import numpy as np
    import torch
    import jax.numpy as jnp
    from scipy.linalg import norm as scipy_norm
    from autograd import numpy as anp

    if interface == "torch":
        if axis is not None:
            axis = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
        return torch.norm(tensor, dim=axis, **kwargs)
    
    elif interface == "jax":
        return jnp.linalg.norm(tensor, axis=axis, **kwargs)
    
    elif interface == "tensorflow":
        import tensorflow as tf
        return tf.norm(tensor, axis=axis, **kwargs)
    
    elif interface == "autograd":
        def _flat_autograd_norm(tensor):
            return anp.sqrt(anp.sum(anp.square(tensor)))
        return _flat_autograd_norm(tensor)
    
    else:  # default to scipy
        return scipy_norm(tensor, axis=axis, **kwargs)

if __name__ == "__main__":
    import torch

    tensor_torch = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    print("Torch Norm:", norm(tensor_torch, interface='torch'))

    tensor_numpy = np.array([[3.0, 4.0], [1.0, 2.0]])
    print("Scipy Norm:", norm(tensor_numpy, interface='scipy'))

    import jax.numpy as jnp
    tensor_jax = jnp.array([[3.0, 4.0], [1.0, 2.0]])
    print("JAX Norm:", norm(tensor_jax, interface='jax'))

    import tensorflow as tf
    tensor_tf = tf.constant([[3.0, 4.0], [1.0, 2.0]])
    print("TensorFlow Norm:", norm(tensor_tf, interface='tensorflow'))

    import autograd.numpy as anp
    tensor_autograd = anp.array([[3.0, 4.0], [1.0, 2.0]])
    print("Autograd Norm:", norm(tensor_autograd, interface='autograd'))
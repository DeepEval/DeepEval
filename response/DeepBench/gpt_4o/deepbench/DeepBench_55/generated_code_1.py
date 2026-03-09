import numpy as np
from scipy.linalg import norm as scipy_norm

def norm(tensor, like="scipy", axis=None, **kwargs):
    if like == "jax":
        import jax.numpy as jnp
        return jnp.linalg.norm(tensor, axis=axis, **kwargs)
    elif like == "tensorflow":
        import tensorflow as tf
        return tf.norm(tensor, axis=axis, **kwargs)
    elif like == "torch":
        import torch
        if axis is not None:
            if isinstance(axis, tuple):
                axis = list(axis)
            return torch.norm(tensor, dim=axis, **kwargs)
        return torch.norm(tensor, **kwargs)
    elif like == "autograd":
        from autograd import numpy as anp
        def _flat_autograd_norm(x):
            return anp.sqrt(anp.sum(x ** 2))
        if len(tensor.shape) == 1:
            return _flat_autograd_norm(tensor)
        else:
            return anp.linalg.norm(tensor, axis=axis, **kwargs)
    else:  # default to "scipy"
        return scipy_norm(tensor, axis=axis, **kwargs)

if __name__ == "__main__":
    # Scipy example
    tensor = np.array([1, 2, 3])
    result = norm(tensor, like="scipy")
    print(f"Scipy norm: {result}")

    # Jax example
    import jax.numpy as jnp
    tensor_jax = jnp.array([1, 2, 3])
    result_jax = norm(tensor_jax, like="jax")
    print(f"Jax norm: {result_jax}")

    # TensorFlow example
    import tensorflow as tf
    tensor_tf = tf.constant([1, 2, 3], dtype=tf.float32)
    result_tf = norm(tensor_tf, like="tensorflow")
    print(f"TensorFlow norm: {result_tf.numpy()}")

    # Torch example
    import torch
    tensor_torch = torch.tensor([1, 2, 3], dtype=torch.float32)
    result_torch = norm(tensor_torch, like="torch")
    print(f"Torch norm: {result_torch}")

    # Autograd example
    from autograd import numpy as anp
    tensor_autograd = anp.array([1, 2, 3])
    result_autograd = norm(tensor_autograd, like="autograd")
    print(f"Autograd norm: {result_autograd}")
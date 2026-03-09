import tensorflow as tf
import numpy as np

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    assert kernel_size >= 3
    assert kernel_size % 2 != 0

    # Create a linear ramp from 0 to 1 to 0
    half_size = (kernel_size // 2) + 1
    ramp_up = np.linspace(0, 1, half_size)
    ramp_down = np.linspace(1, 0, half_size - 1)
    
    # Combine the ramps to form a triangular shape
    kernel_array = np.concatenate([ramp_up, ramp_down])

    # Convert the numpy array to a TensorFlow tensor
    kernel_weights = tf.convert_to_tensor(kernel_array, dtype=tf.float32)
    
    return kernel_weights

if __name__ == "__main__":
    kernel_size = 5  # Example with an odd kernel size
    kernel = triangular_kernel1d(kernel_size)
    print("Triangular kernel weights for kernel size", kernel_size, ":", kernel.numpy())
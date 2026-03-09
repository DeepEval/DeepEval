import tensorflow as tf
from typing import Optional

class TensorType(tf.Tensor):
    pass

class Kernel:
    def __init__(self, weight_variances: tf.Tensor, bias_variance: tf.Tensor):
        self.weight_variances = weight_variances
        self.bias_variance = bias_variance

class ArcCosine(Kernel):
    def _full_weighted_product(self, X: TensorType, X2: Optional[TensorType]) -> tf.Tensor:
        if X2 is None:
            X2 = X

        product = tf.matmul(X, X2, transpose_b=True)
        weighted_product = self.weight_variances * product + self.bias_variance

        return weighted_product

if __name__ == "__main__":
    # Example usage
    weight_variances = tf.constant([2.0], dtype=tf.float32)
    bias_variance = tf.constant(1.0, dtype=tf.float32)
    
    arc_cosine_kernel = ArcCosine(weight_variances, bias_variance)
    
    X = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    X2 = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
    
    result = arc_cosine_kernel._full_weighted_product(X, X2)
    print("Weighted product with X and X2:")
    print(result.numpy())
    
    result_self = arc_cosine_kernel._full_weighted_product(X, None)
    print("Weighted product with X and itself:")
    print(result_self.numpy())
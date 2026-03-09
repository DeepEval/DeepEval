import tensorflow as tf
from typing import Optional

class Kernel:
    def __init__(self, weight_variances, bias_variance):
        self.weight_variances = weight_variances
        self.bias_variance = bias_variance

class ArcCosine(Kernel):
    def _full_weighted_product(self, X: tf.Tensor, X2: Optional[tf.Tensor]) -> tf.Tensor:
        if X2 is None:
            X2 = X
        
        weighted_product = tf.matmul(X, X2) * self.weight_variances
        weighted_product = weighted_product + self.bias_variance
        return weighted_product

if __name__ == "__main__":
    weight_variances = tf.constant(0.5)
    bias_variance = tf.constant(1.0)

    arc_cosine_kernel = ArcCosine(weight_variances, bias_variance)

    X = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    X2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])

    result = arc_cosine_kernel._full_weighted_product(X, X2)
    print("Weighted product with X2:", result.numpy())

    result_self_product = arc_cosine_kernel._full_weighted_product(X, None)
    print("Weighted product with itself:", result_self_product.numpy())